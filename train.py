#!/usr/bin/env python3
"""Part 4 — train ScratchCNN and TransferModel; package artifacts + signature."""

from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
)

from leaffliction.cli import console, die
from leaffliction.dataset import LeafDataset
from leaffliction.models import ScratchCNN, TransferModel
from leaffliction.predictor import ImageNetMean, ImageNetStd
from leaffliction.seed import set_seed
from leaffliction.signature import write_signature
from leaffliction.trainer import TrainConfig, save_state, train
from leaffliction.viz import confusion_matrix_plot, learning_curves

app = typer.Typer(add_completion=False, help=__doc__)


def _build_loaders(directory: Path, split: float, batch: int, seed: int):
    """Build DataLoaders.

    Two LeafDataset instances over the same directory are created so that train and val
    can have different transforms (augmentation only on train). discover_classes returns
    sorted output, so the two instances iterate samples in identical order — making the
    integer indices from train_test_split consistent across both.
    """
    # Resize first so any input size works (PDF data is 256x256 but evaluator
    # may bring different sizes); then augment + normalize.
    train_tf = Compose(
        [
            Resize((256, 256), antialias=True),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            Normalize(mean=ImageNetMean, std=ImageNetStd),
        ]
    )
    val_tf = Compose(
        [
            Resize((256, 256), antialias=True),
            Normalize(mean=ImageNetMean, std=ImageNetStd),
        ]
    )

    train_full = LeafDataset(directory, transform=train_tf)
    val_full = LeafDataset(directory, transform=val_tf)
    labels = [lab for _, lab in train_full.samples]

    train_idx, val_idx = train_test_split(
        list(range(len(labels))),
        test_size=1 - split,
        stratify=labels,
        random_state=seed,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_labels = np.array([labels[i] for i in train_idx])
    class_count = np.bincount(train_labels)
    sample_weights = 1.0 / class_count[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # num_workers=8 + persistent_workers=True keeps the worker processes
    # alive across epochs, which on macOS shaves a large amount of overhead
    # (each fork() + module-import is expensive).
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        sampler=sampler,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_loader, val_loader, train_full.classes, train_full.class_to_idx


def _train_one(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> dict:
    """Train one model, printing progress per epoch so the log file is readable."""
    start = time.time()
    console.print(f"[info]>>> Starting {name} ({cfg.epochs} epochs max)...[/info]")

    def on_epoch_end(epoch: int, metrics: dict) -> None:
        elapsed = time.time() - start
        console.print(
            f"  {name} epoch {epoch:3d}/{cfg.epochs} "
            f"| train: loss={metrics['train_loss']:.4f} acc={metrics['train_acc']:.4f} "
            f"| val: loss={metrics['val_loss']:.4f} acc={metrics['val_acc']:.4f} "
            f"| t={elapsed:.0f}s"
        )

    result = train(model, train_loader, val_loader, cfg, on_epoch_end=on_epoch_end)
    total = time.time() - start
    console.print(
        f"[ok]<<< {name} done in {total:.0f}s. "
        f"Best val_acc={result.best_val_acc:.4f} @ epoch {result.best_epoch}[/ok]"
    )
    return {
        "history": result.history,
        "best_val_acc": result.best_val_acc,
        "best_epoch": result.best_epoch,
        "state": result.state_dict,
    }


_DIRECTORY = typer.Argument(..., exists=True)
_MODEL = typer.Option("both", "--model")
_EPOCHS = typer.Option(25, "--epochs")
_BATCH = typer.Option(32, "--batch")
_SEED = typer.Option(42, "--seed")
_SPLIT = typer.Option(0.8, "--split")
_OUT = typer.Option(Path("artifacts"), "--out")


@app.command()
def main(
    directory: Path = _DIRECTORY,
    model: str = _MODEL,
    epochs: int = _EPOCHS,
    batch: int = _BATCH,
    seed: int = _SEED,
    split: float = _SPLIT,
    out: Path = _OUT,
) -> None:
    if model not in ("both", "scratch", "transfer"):
        die(f"Invalid --model: {model}. Choose from both/scratch/transfer.")

    set_seed(seed)
    console.print(f"[info]Loading dataset from {directory} ...[/info]")
    train_loader, val_loader, classes, class_to_idx = _build_loaders(directory, split, batch, seed)
    n_classes = len(classes)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset)  # type: ignore[arg-type]
    console.print(f"[info]{n_train} train + {n_val} val, {n_classes} classes[/info]")
    if n_val < 100:
        die("Validation set has fewer than 100 images — PDF requires >= 100.")

    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict] = {}

    if model in ("both", "scratch"):
        console.print("[info]Training ScratchCNN ...[/info]")
        artifacts["scratch"] = _train_one(
            "scratch",
            ScratchCNN(num_classes=n_classes),
            train_loader,
            val_loader,
            TrainConfig(epochs=epochs, lr=1e-3, patience=5),
        )
        save_state(artifacts["scratch"]["state"], out / "model_scratch.pt")

    if model in ("both", "transfer"):
        console.print("[info]Training TransferModel ...[/info]")
        artifacts["transfer"] = _train_one(
            "transfer",
            TransferModel(num_classes=n_classes, pretrained=True),
            train_loader,
            val_loader,
            TrainConfig(epochs=epochs, lr=1e-3, patience=5, unfreeze_after=5),
        )
        save_state(artifacts["transfer"]["state"], out / "model_transfer.pt")

    best_name = max(artifacts, key=lambda k: artifacts[k]["best_val_acc"])
    learning_curves(artifacts[best_name]["history"], out / "learning_curves.png")

    best_model = (
        ScratchCNN(num_classes=n_classes)
        if best_name == "scratch"
        else TransferModel(num_classes=n_classes, pretrained=False)
    )
    best_model.load_state_dict(artifacts[best_name]["state"])
    best_model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            logits = best_model(x)
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    confusion_matrix_plot(cm, classes, out / "confusion_matrix.png")
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=4)
    )

    metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "image_size": 256,
        "normalize_mean": ImageNetMean,
        "normalize_std": ImageNetStd,
        "models": {
            name: {
                "val_accuracy": round(art["best_val_acc"], 4),
                "best_epoch": art["best_epoch"],
            }
            for name, art in artifacts.items()
        },
        "split": {"train": split, "val": round(1 - split, 4)},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    models_zip = Path("trained_models.zip")
    with zipfile.ZipFile(models_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir():
            zf.write(p, arcname=p.name)

    # augmented_directory.zip is produced by Augmentation.py (--balance). If it
    # exists alongside trained_models.zip, include both in signature.txt.
    aug_zip = Path("augmented_directory.zip")
    sig_inputs = [models_zip] + ([aug_zip] if aug_zip.exists() else [])
    write_signature(sig_inputs, Path("signature.txt"))
    console.print("[ok]Done. trained_models.zip + signature.txt generated.[/ok]")
    for name, art in artifacts.items():
        console.print(
            f"  [info]{name}[/info]: best val_acc={art['best_val_acc']:.4f} "
            f"@ epoch {art['best_epoch']}"
        )


if __name__ == "__main__":
    app()
