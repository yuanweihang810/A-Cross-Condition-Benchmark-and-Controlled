# -*- coding: utf-8 -*-
"""SE / CBAM baselines inserted into the residual branch.

These are parameter-adding, gate-based baselines.  They are here to test whether
TPR's parameter-free geometry is competitive with standard attention modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpr import _residual_of_block


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * w
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        return x * s


class AttentionWrapper(nn.Module):
    def __init__(self, block: nn.Module, kind: str):
        super().__init__()
        self.block = block
        if hasattr(block, "conv3"):
            c = block.conv3.out_channels
        else:
            c = block.conv2.out_channels
        if kind == "se":
            self.attn = SEBlock(c)
        elif kind == "cbam":
            self.attn = CBAMBlock(c)
        else:
            raise ValueError(kind)

    def forward(self, x):
        identity = x
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        out = _residual_of_block(self.block, x)
        out = self.attn(out)
        out = out + identity
        out = self.block.relu(out)
        return out


def convert_to_attention(model: nn.Module, kind: str = "se") -> nn.Module:
    for name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, name)
        for i, block in enumerate(layer):
            layer[i] = AttentionWrapper(block, kind)
    return model


__all__ = ["SEBlock", "CBAMBlock", "AttentionWrapper", "convert_to_attention"]
