# -*- coding: utf-8 -*-
"""
Tangent-Path Residual (TPR) for ResNet-18.

Core idea
---------
Standard residual:  y = x + F(x)                  (one Euler point, ambient space)
TPR:                u0 = P_x^T F(x)
                    x1 = x + h u0
                    u1 = P_x1^T F(x1)
                    T  = 0.5 (u0 + u1)             (trapezoidal path integral)
                    y  = x + h P_x^T T

P_x^T is a *parameter-free* global tangent projection: it removes the
component of the residual whose global mean is parallel to the global mean
of the current feature.  It is a geometric constraint, not a learned gate.

The same residual function F (same conv/BN weights) is evaluated twice.
No new parameters are introduced (unless learn_h is used).

This file is deliberately dependency-light: torch + torchvision only.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ProjectionMode = Literal["none", "global", "mean", "local"]


# ---------------------------------------------------------------------------
# 1. Tangent projections
# ---------------------------------------------------------------------------
def tangent_project(
    f: torch.Tensor,
    x: torch.Tensor,
    mode: ProjectionMode = "global",
    rho: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Remove the radial component of f relative to x.

    Args:
        f: residual feature map [B, C, H, W].
        x: reference feature map [B, C, H, W] (same shape).
        mode:
            "none"   : identity (standard residual / Heun baseline).
            "global" : project the *global mean* onto the tangent space of
                       the global mean of x.  One scalar coefficient per sample.
            "local"  : per-pixel projection (more aggressive; useful ablation).
        rho: partial projection strength in [0, 1].  rho=1 is full projection,
             rho=0 is no projection.  Fixed hyperparameter, not a gate.
        eps: numerical floor.

    Returns:
        f_perp with the same shape as f.

    Notes:
        For "global" the operator is idempotent: applying it twice gives the
        same result, because the global mean of the output is orthogonal to the
        global mean of x.
    """
    if mode == "none" or rho == 0.0:
        return f

    if mode == "mean":
        # Project only the global mean vector, then apply the resulting
        # constant channel shift to the whole map.  This preserves spatial
        # residuals much better than subtracting alpha * x.
        gx = x.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
        gf = f.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
        denom = (gx * gx).sum(dim=1, keepdim=True).clamp_min(eps)
        alpha = (gf * gx).sum(dim=1, keepdim=True) / denom
        return f - (rho * alpha) * gx

    if mode == "global":
        gx = x.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
        gf = f.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
        denom = (gx * gx).sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,1,1]
        alpha = (gf * gx).sum(dim=1, keepdim=True) / denom         # [B,1,1,1]
        return f - (rho * alpha) * x

    if mode == "local":
        denom = (x * x).sum(dim=1, keepdim=True).clamp_min(eps)    # [B,1,H,W]
        alpha = (f * x).sum(dim=1, keepdim=True) / denom           # [B,1,H,W]
        return f - (rho * alpha) * x

    raise ValueError(f"unknown projection mode: {mode}")


# ---------------------------------------------------------------------------
# 2. Residual function of a torchvision BasicBlock / Bottleneck
# ---------------------------------------------------------------------------
def _residual_of_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run only the residual branch F(x) of a torchvision ResNet block.

    Supports BasicBlock and Bottleneck.  The block's downsample is *not*
    applied here; the caller handles the identity path.
    """
    out = block.conv1(x)
    out = block.bn1(out)
    out = block.relu(out)

    if hasattr(block, "conv2") and hasattr(block, "bn2"):
        out = block.conv2(out)
        out = block.bn2(out)

    # Bottleneck has conv3/bn3
    if hasattr(block, "conv3") and hasattr(block, "bn3"):
        out = block.relu(out)
        out = block.conv3(out)
        out = block.bn3(out)

    return out

def _bn_modules(block: nn.Module):
    return [m for m in block.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _bn_eval_mode(block: nn.Module):
    """Put BN layers in eval mode for the probe pass.

    The probe F(x_t) must not update the shared BN running statistics: doing so
    corrupts the eval-time statistics and makes the model collapse after the
    TPR warm-up (this happened with the naive Heun baseline).
    """
    mods = _bn_modules(block)
    states = [m.training for m in mods]
    for m in mods:
        m.eval()
    return mods, states


def _bn_restore(block: nn.Module, mods, states) -> None:
    del block
    for m, s in zip(mods, states):
        m.train(s)


# ---------------------------------------------------------------------------
# 3. TPR wrapper
# ---------------------------------------------------------------------------
class TPRBlock(nn.Module):
    """Wrap an existing torchvision residual block with a Tangent-Path update.

    The wrapped block keeps its original weights and state_dict names under
    self.block.  The wrapper adds no learnable parameters unless
    learn_h=True (one scalar per wrapped block).

    Only shape-preserving blocks should be wrapped.  Blocks with a downsample
    shortcut are left as standard blocks by convert_to_tpr, because the
    residual function is defined on the pre-downsample input.
    """

    def __init__(
        self,
        block: nn.Module,
        h: float = 1.0,
        projection: ProjectionMode = "global",
        rho: float = 1.0,
        K: int = 2,
        final_projection: bool = True,
        learn_h: bool = False,
    ) -> None:
        super().__init__()
        if K not in (1, 2, 3):
            raise ValueError("K must be 1, 2, or 3")
        self.block = block
        self.projection = projection
        self.rho = float(rho)
        self.K = int(K)
        self.final_projection = bool(final_projection)

        if learn_h:
            self.h = nn.Parameter(torch.tensor(float(h)))
        else:
            self.register_buffer("h", torch.tensor(float(h)))

        # Training schedule: 0 = standard residual, 1 = full TPR.
        # This is a buffer, not a learned gate.
        self.register_buffer("gamma", torch.tensor(1.0))

    def set_gamma(self, gamma: float) -> None:
        self.gamma.fill_(float(gamma))

    def _quadrature_nodes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Nodes on [0, 1]; weights sum to 1.
        if self.K == 1:
            nodes = [0.0]
            weights = [1.0]
        elif self.K == 2:
            nodes = [0.0, 1.0]
            weights = [0.5, 0.5]
        else:  # Simpson
            nodes = [0.0, 0.5, 1.0]
            weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]
        return (
            torch.tensor(nodes, dtype=torch.float32, device=self.h.device),
            torch.tensor(weights, dtype=torch.float32, device=self.h.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.block.downsample is not None:
            # Defensive: convert_to_tpr should not wrap these.
            identity = self.block.downsample(x)

        out = _residual_of_block(self.block, x)  # F(x)

        if self.gamma.item() > 0.0 and self.block.downsample is None:
            h = self.h
            u0 = tangent_project(out, identity, self.projection, self.rho)

            nodes, weights = self._quadrature_nodes()
            T = torch.zeros_like(u0)
            for t, w in zip(nodes.tolist(), weights.tolist()):
                if t == 0.0:
                    u_t = u0
                else:
                    x_t = identity + (h * t) * u0
                    bn_mods, bn_states = _bn_eval_mode(self.block)
                    try:
                        f_t = _residual_of_block(self.block, x_t)
                    finally:
                        _bn_restore(self.block, bn_mods, bn_states)
                    u_t = tangent_project(f_t, x_t, self.projection, self.rho)
                T = T + w * u_t

            if self.final_projection:
                T = tangent_project(T, identity, self.projection, self.rho)

            tpr_update = h * T
            # gamma=0 -> exact standard residual; gamma=1 -> full TPR.
            out = (1.0 - self.gamma) * out + self.gamma * tpr_update

        out = out + identity
        out = self.block.relu(out)
        return out


# ---------------------------------------------------------------------------
# 4. Conversion utility
# ---------------------------------------------------------------------------
def convert_to_tpr(
    model: nn.Module,
    stages: Iterable[int] = (3, 4),
    h: float = 1.0,
    projection: ProjectionMode = "global",
    rho: float = 1.0,
    K: int = 2,
    final_projection: bool = True,
    learn_h: bool = False,
) -> nn.Module:
    """Replace shape-preserving blocks in model.layerN with TPRBlock.

    Args:
        model: torchvision ResNet (or compatible, with layer1..layer4).
        stages: which stages to convert, e.g. (3, 4) or (4,).
        h, projection, rho, K, final_projection, learn_h: TPR options.

    Returns:
        The same model (in place) for convenience.
    """
    for stage in stages:
        layer = getattr(model, f"layer{stage}", None)
        if layer is None:
            raise ValueError(f"model has no layer{stage}")
        for i, block in enumerate(layer):
            if getattr(block, "downsample", None) is not None:
                continue  # keep transition blocks standard
            layer[i] = TPRBlock(
                block,
                h=h,
                projection=projection,
                rho=rho,
                K=K,
                final_projection=final_projection,
                learn_h=learn_h,
            )
    return model


def set_tpr_gamma(model: nn.Module, gamma: float) -> None:
    """Set the warm-up coefficient of every TPRBlock in model."""
    for m in model.modules():
        if isinstance(m, TPRBlock):
            m.set_gamma(gamma)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 5. Convenience builder
# ---------------------------------------------------------------------------
def tpr_resnet18(
    num_classes: int = 7,
    pretrained: bool = False,
    stages: Iterable[int] = (3, 4),
    **tpr_kwargs,
) -> nn.Module:
    """Build a torchvision ResNet-18 and convert selected stages to TPR."""
    import torchvision.models as tvm

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    convert_to_tpr(model, stages=stages, **tpr_kwargs)
    return model


__all__ = [
    "tangent_project",
    "TPRBlock",
    "convert_to_tpr",
    "set_tpr_gamma",
    "count_parameters",
    "tpr_resnet18",
]
