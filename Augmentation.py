#!/usr/bin/env python3
"""Part 2 — augmentation: 6 ops on a single image, or balance a directory."""

from __future__ import annotations

from pathlib import Path

import typer

from leaffliction.augment import (
    AUGMENTATION_OPS,
    apply_op,
    balance_directory,
    load_image,
    save_with_suffix,
)
from leaffliction.cli import console, die
from leaffliction.viz import grid

app = typer.Typer(add_completion=False, help=__doc__)

_TARGET = typer.Argument(
    ..., exists=True, help="Image file (single mode) or directory (batch mode)."
)
_BALANCE = typer.Option(False, "--balance", help="Batch mode: balance classes via augmentation.")
_OUTPUT = typer.Option(
    Path("augmented_directory"), "--output", "-o", help="Batch mode output dir."
)
_TARGET_COUNT = typer.Option(
    None, "--target-count", help="Per-class target count (default: max class)."
)
_SEED = typer.Option(42, "--seed", help="Random seed.")


@app.command()
def main(
    target: Path = _TARGET,
    balance: bool = _BALANCE,
    output: Path = _OUTPUT,
    target_count: int | None = _TARGET_COUNT,
    seed: int = _SEED,
) -> None:
    if target.is_file():
        rgb = load_image(target)
        outputs = [("Original", rgb)]
        for name in AUGMENTATION_OPS:
            aug = apply_op(name, rgb)
            save_with_suffix(target, aug, name)
            outputs.append((name, aug))
        console.print(f"[ok]Saved 6 augmentations next to {target.name}[/ok]")
        grid(outputs)
        return

    if not balance:
        die("Directory passed without --balance. Pass --balance to run batch mode.")

    console.print(f"[info]Balancing {target} into {output} ...[/info]")
    summary = balance_directory(target, output, target_count=target_count, seed=seed)
    for cls, n in summary.items():
        console.print(f"  [info]{cls}[/info]: {n} images")
    console.print(f"[ok]Done. Output at {output}[/ok]")


if __name__ == "__main__":
    app()
