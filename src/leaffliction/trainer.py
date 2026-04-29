"""Shared training loop with early stopping + LR plateau scheduler."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-3
    unfreeze_after: int | None = None  # for transfer model: epoch at which to unfreeze


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
    )
    best_val_acc: float = 0.0
    best_epoch: int = 0
    state_dict: dict[str, Any] | None = None


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += float(loss) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    device = device_auto()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    result = TrainResult()
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        if (
            config.unfreeze_after is not None
            and epoch == config.unfreeze_after + 1
            and hasattr(model, "unfreeze")
        ):
            model.unfreeze()
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr * 0.1,
                weight_decay=config.weight_decay,
            )

        train_loss, train_acc = _epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, device)

        result.history["train_loss"].append(train_loss)
        result.history["val_loss"].append(val_loss)
        result.history["train_acc"].append(train_acc)
        result.history["val_acc"].append(val_acc)

        scheduler.step(val_acc)
        if val_acc > result.best_val_acc + config.min_delta:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if on_epoch_end is not None:
            on_epoch_end(
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                },
            )

        if epochs_no_improve >= config.patience:
            break

    result.state_dict = best_state
    return result


def save_state(state: dict[str, Any], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
