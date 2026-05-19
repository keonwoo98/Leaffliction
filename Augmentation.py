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
    ...,
    exists=True,
    help="Image file (-> single mode) or directory (-> batch balance mode).",
)
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
    output: Path = _OUTPUT,
    target_count: int | None = _TARGET_COUNT,
    seed: int = _SEED,
) -> None:
    # Single-image mode: file argument
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

    # Batch mode: directory argument (auto)
    if not target.is_dir():
        die(f"Target is neither a file nor a directory: {target}")

    console.print(f"[info]Balancing {target} into {output} ...[/info]")
    summary = balance_directory(target, output, target_count=target_count, seed=seed)
    for cls, n in summary.items():
        console.print(f"  [info]{cls}[/info]: {n} images")
    console.print(f"[ok]Done. Output at {output}[/ok]")


if __name__ == "__main__":
    app()
