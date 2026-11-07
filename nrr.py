# -*- coding: utf-8 -*-
"""Neutral-Reference Residual (NRR).

Expression is a change from a neutral/low-frequency face.  NRR subtracts the
residual computed on a blurred reference from the residual on the original:

    y = x + F(x) - lambda * F(blur(x))

The blurred reference is a cheap neutral proxy.  The block reuses the TPRBlock
gamma buffer so the existing training schedule works unchanged.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpr import TPRBlock, _bn_eval_mode, _bn_restore, _residual_of_block


class NRRBlock(TPRBlock):
    def __init__(self, block: nn.Module, lam: float = 0.5, blur_kernel: int = 5):
        super().__init__(block, h=1.0, projection="none", K=2,
                         final_projection=False)
        self.lam = float(lam)
        self.blur_kernel = int(blur_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out = _residual_of_block(self.block, x)

        if self.gamma.item() > 0.0 and self.block.downsample is None:
            k = self.blur_kernel
            x_ref = F.avg_pool2d(identity, kernel_size=k, stride=1, padding=k // 2)
            bn_mods, bn_states = _bn_eval_mode(self.block)
            try:
                f_ref = _residual_of_block(self.block, x_ref)
            finally:
                _bn_restore(self.block, bn_mods, bn_states)
            nrr_update = out - self.lam * f_ref
            out = (1.0 - self.gamma) * out + self.gamma * nrr_update

        out = out + identity
        out = self.block.relu(out)
        return out


def convert_to_nrr(model: nn.Module, stages: Iterable[int] = (3, 4),
                   lam: float = 0.5, blur_kernel: int = 5) -> nn.Module:
    for stage in stages:
        layer = getattr(model, f"layer{stage}", None)
        if layer is None:
            raise ValueError(f"model has no layer{stage}")
        for i, block in enumerate(layer):
            if getattr(block, "downsample", None) is not None:
                continue
            layer[i] = NRRBlock(block, lam=lam, blur_kernel=blur_kernel)
    return model


__all__ = ["NRRBlock", "convert_to_nrr"]
