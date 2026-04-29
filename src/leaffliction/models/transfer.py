"""EfficientNet-B0 transfer learning with two-stage fine-tuning support."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class TransferModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone
        self.freeze()

    def freeze(self) -> None:
        """Freeze the backbone — only the new classifier head is trainable."""
        for name, p in self.model.named_parameters():
            p.requires_grad = "classifier" in name

    def unfreeze(self) -> None:
        """Unfreeze the entire model for fine-tuning."""
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
