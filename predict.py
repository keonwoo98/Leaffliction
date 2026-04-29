#!/usr/bin/env python3
"""Part 4 — predict disease class for a single leaf image."""
from __future__ import annotations

from pathlib import Path

import typer

from leaffliction.cli import console, die
from leaffliction.predictor import predict, render

app = typer.Typer(add_completion=False, help=__doc__)

_IMAGE = typer.Argument(..., exists=True, help="Image to classify.")
_ZIP = typer.Option(
    Path("trained_models.zip"),
    "--zip",
    help="Trained models archive (default: trained_models.zip).",
)
_MODEL = typer.Option(
    "transfer",
    "--model",
    help="Preferred model: 'transfer' or 'scratch'.",
)
_SAVE = typer.Option(None, "--save", help="Save figure to PNG instead of showing.")


@app.command()
def main(
    image: Path = _IMAGE,
    zip_path: Path = _ZIP,
    model: str = _MODEL,
    save: Path | None = _SAVE,
) -> None:
    if not zip_path.exists():
        die(f"Zip not found: {zip_path}. Run train.py first.")
    result = predict(image, zip_path, prefer=model)
    console.print(
        f"[ok]Class predicted: {result['class']} "
        f"({result['confidence']:.1%}) [model={result['model_used']}][/ok]"
    )
    render(result, save=save)


if __name__ == "__main__":
    app()
