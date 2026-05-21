#!/usr/bin/env python3
"""Evaluate a trained model on the deterministic validation split.

Reproduces the stratified 80/20 split used by train.py (seed=42 by
default), loads weights from trained_models.zip, runs forward-only on
the ~1,445 held-out images, and prints overall + per-class accuracy.

Use this on defense day to demonstrate the PDF requirement "run on
>=100 images, accuracy must be >= 90%" without trusting the number
printed by train.py — the val set is rebuilt here from scratch so the
evaluator can see it is the same images held out during training.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
import typer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.v2 import Compose, Normalize, Resize

from leaffliction.cli import console, die
from leaffliction.dataset import LeafDataset
from leaffliction.models import ScratchCNN, TransferModel
from leaffliction.predictor import ImageNetMean, ImageNetStd

app = typer.Typer(add_completion=False, help=__doc__)


def _build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "scratch":
        return ScratchCNN(num_classes=num_classes)
    if name == "transfer":
        return TransferModel(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model: {name}")


_DIRECTORY = typer.Argument(
    Path("images"),
    exists=True,
    help="Raw image dataset root (same one train.py used).",
)
_ZIP = typer.Option(Path("trained_models.zip"), "--zip", help="Trained models archive.")
_MODEL_NAME = typer.Option(
    "scratch", "--model", help="Which weights to evaluate: scratch / transfer."
)
_SEED = typer.Option(42, "--seed", help="Random state used for the split (must match train.py).")
_SPLIT = typer.Option(0.8, "--split", help="Train fraction (must match train.py).")
_BATCH = typer.Option(32, "--batch")


@app.command()
def main(
    directory: Path = _DIRECTORY,
    zip_path: Path = _ZIP,
    model_name: str = _MODEL_NAME,
    seed: int = _SEED,
    split: float = _SPLIT,
    batch: int = _BATCH,
) -> None:
    if not zip_path.exists():
        die(f"{zip_path} not found. Run train.py first.")

    # 1) Unpack trained_models.zip alongside it.
    extract_dir = zip_path.parent / f".{zip_path.stem}_unpacked"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    metadata = json.loads((extract_dir / "metadata.json").read_text())
    classes: list[str] = metadata["classes"]

    # 2) Rebuild the dataset + val transform.
    val_tf = Compose(
        [
            Resize((256, 256), antialias=True),
            Normalize(mean=ImageNetMean, std=ImageNetStd),
        ]
    )
    full = LeafDataset(directory, transform=val_tf)
    if full.classes != classes:
        die(
            f"Class layout mismatch.\n"
            f"  metadata.json: {classes}\n"
            f"  {directory}:   {full.classes}"
        )

    # 3) Reproduce the train/val split deterministically.
    labels = [lab for _, lab in full.samples]
    _, val_idx = train_test_split(
        list(range(len(labels))),
        test_size=1 - split,
        stratify=labels,
        random_state=seed,
    )
    val_ds = Subset(full, val_idx)
    n_val = len(val_ds)
    console.print(
        f"[info]Evaluating model='{model_name}' on {n_val} held-out images "
        f"(seed={seed}, split={split})...[/info]"
    )

    # 4) Load weights.
    weight_file = extract_dir / f"model_{model_name}.pt"
    if not weight_file.exists():
        die(f"{weight_file.name} not in {zip_path}")
    model = _build_model(model_name, num_classes=len(classes))
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()

    # 5) Forward pass over val set, tally correct/total + per-class.
    loader = DataLoader(val_ds, batch_size=batch, num_workers=4)
    correct, total = 0, 0
    per_class: dict[str, list[int]] = {c: [0, 0] for c in classes}
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = logits.argmax(1)
            for p, t in zip(preds.tolist(), y.tolist(), strict=True):
                per_class[classes[t]][1] += 1
                total += 1
                if p == t:
                    correct += 1
                    per_class[classes[t]][0] += 1

    # 6) Verdict.
    acc = correct / total if total else 0.0
    console.print(f"[ok]Overall: {correct}/{total} = {acc:.4f} ({acc:.2%})[/ok]")
    if total < 100:
        console.print(f"[warn]Warning: only {total} images (< 100, PDF minimum)[/warn]")
    if acc >= 0.90:
        console.print(f"[ok]PASS — PDF requirement >= 90% cleared by {(acc - 0.9):.4f}[/ok]")
    else:
        console.print(f"[err]FAIL — below PDF threshold ({acc:.4f} < 0.90)[/err]")

    console.print("\n[info]Per-class breakdown:[/info]")
    for c in classes:
        ok, tot = per_class[c]
        rate = ok / tot if tot else 0.0
        marker = "OK " if rate >= 0.90 else "low"
        console.print(f"  {marker}  {c:18s} {ok:4d}/{tot:4d}  ({rate:.2%})")


if __name__ == "__main__":
    app()
