"""Stub — real implementation in Task 11."""
from __future__ import annotations

from torch import nn


class TransferModel(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x):  # pragma: no cover
        raise NotImplementedError("Implemented in Task 11")
