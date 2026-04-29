"""Hand-designed CNN — every layer is the author's. Defense-friendly."""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
