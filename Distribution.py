#!/usr/bin/env python3
"""Part 1 — display class distribution as pie + bar charts."""
from __future__ import annotations

from pathlib import Path

import typer

from leaffliction.cli import console, die
from leaffliction.dataset import discover_classes
from leaffliction.viz import pie_and_bar

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    directory: Path = typer.Argument(  # noqa: B008
        ..., exists=True, help="Path to dataset root."
    ),
    save: Path | None = typer.Option(  # noqa: B008
        None, "--save", help="If set, save chart PNG instead of showing."
    ),
) -> None:
    try:
        classes = discover_classes(directory)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        die(str(exc))
    counts = {name: len(paths) for name, paths in classes.items()}
    title = directory.name or directory.resolve().name
    console.print(
        f"[info]Found {len(counts)} classes, "
        f"{sum(counts.values())} images in {directory}[/info]"
    )
    pie_and_bar(counts, title=title, save=save)


if __name__ == "__main__":
    app()
