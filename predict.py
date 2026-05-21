#!/usr/bin/env python3
"""Part 4 — predict disease class for one or many leaf images.

Usage modes (auto-dispatch based on arguments):
- Single file  → ./predict.py img.JPG          : console line + matplotlib figure
- Many files   → ./predict.py img1.JPG img2.JPG: console table only
- Directory    → ./predict.py images/Apple_rust: globs *.JPG inside, console table
- Save mode    → add `--save out/`              : also writes <stem>_pred.png per image
"""

from __future__ import annotations

from pathlib import Path

import typer

from leaffliction.cli import console, die
from leaffliction.predictor import load_artifact, predict_one, render

app = typer.Typer(add_completion=False, help=__doc__)

_PATHS = typer.Argument(
    ...,
    exists=True,
    help="One or more image paths, or a directory containing *.JPG.",
)
_ZIP = typer.Option(
    Path("trained_models.zip"),
    "--zip",
    help="Trained models archive (default: trained_models.zip).",
)
_MODEL = typer.Option(
    "scratch",
    "--model",
    help="Preferred model: 'scratch' (default, hand-designed CNN) or 'transfer'.",
)
_SAVE = typer.Option(
    None,
    "--save",
    help="Single mode: save figure as PNG. Multi mode: directory to dump <stem>_pred.png files.",
)


def _expand(paths: list[Path]) -> list[Path]:
    """Expand directories into their *.JPG contents; keep files as-is."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            jpgs = sorted(p.rglob("*.JPG")) + sorted(p.rglob("*.jpg"))
            if not jpgs:
                die(f"No *.JPG inside {p}")
            out.extend(jpgs)
        else:
            out.append(p)
    return out


@app.command()
def main(
    paths: list[Path] = _PATHS,
    zip_path: Path = _ZIP,
    model: str = _MODEL,
    save: Path | None = _SAVE,
) -> None:
    if not zip_path.exists():
        die(f"Zip not found: {zip_path}. Run train.py first.")

    expanded = _expand(paths)

    # ── Single-image mode (back-compat with PDF example) ─────────────
    if len(expanded) == 1:
        artifact = load_artifact(zip_path, prefer=model)
        result = predict_one(artifact, expanded[0])
        console.print(
            f"[ok]Class predicted: {result['class']} "
            f"({result['confidence']:.1%}) [model={result['model_used']}][/ok]"
        )
        render(result, save=save)
        return

    # ── Multi-image mode (folder / multiple args) ────────────────────
    artifact = load_artifact(zip_path, prefer=model)
    console.print(
        f"[info]Predicting {len(expanded)} images with model={artifact.model_used}...[/info]"
    )

    save_dir: Path | None = None
    if save is not None:
        save_dir = save
        save_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    checkable = 0
    for path in expanded:
        result = predict_one(artifact, path)
        # If the filename starts with a known class name we can self-check.
        # (Defense day: evaluator may rename files to prevent this — still fine,
        # we just skip the check and print the prediction.)
        true_cls = _guess_class_from_name(path.name, artifact.classes)
        if true_cls is not None:
            checkable += 1
            mark = "OK " if true_cls == result["class"] else "ERR"
            if true_cls == result["class"]:
                correct += 1
        else:
            mark = "   "
        console.print(
            f"  {mark}  {result['class']:18s} ({result['confidence']:5.1%})  ← {path.name}"
        )
        if save_dir is not None:
            render(result, save=save_dir / f"{path.stem}_pred.png")

    if checkable > 0:
        rate = correct / checkable
        console.print(
            f"\n[ok]Self-check: {correct}/{checkable} = {rate:.2%}[/ok]"
            f" [info](only filenames containing a known class were checked)[/info]"
        )


def _guess_class_from_name(name: str, classes: list[str]) -> str | None:
    """Return the longest class label whose name is a case-insensitive prefix match.

    Lets defense day Unit_test1/2 (e.g. `Apple_healthy1.JPG`) self-check, but
    silently skips when the evaluator renamed files to prevent cheating.
    """
    stem = name.lower()
    # Match longest class first so e.g. "Apple_Black_rot" wins over "Apple".
    for cls in sorted(classes, key=len, reverse=True):
        if stem.startswith(cls.lower()):
            return cls
    return None


if __name__ == "__main__":
    app()
