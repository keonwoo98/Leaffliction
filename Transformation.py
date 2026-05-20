#!/usr/bin/env python3
"""Part 3 — plantCV transformations and color histogram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import typer
from PIL import Image
from rich.progress import track

from leaffliction.cli import console, die
from leaffliction.transform import all_transforms, color_histogram, load_rgb

app = typer.Typer(
    add_completion=False,
    help=__doc__,
    context_settings={"help_option_names": ["-h", "--help"]},
)

FLAG_TO_KEY = {
    "blur": "GaussianBlur",
    "mask": "Mask",
    "roi": "RoiObjects",
    "analyze": "AnalyzeObject",
    "landmarks": "Pseudolandmarks",
}

_IMAGE = typer.Argument(None, help="Single image path (display mode).")
_SRC = typer.Option(None, "-src", "--src", help="Source directory (batch mode).")
_DST = typer.Option(None, "-dst", "--dst", help="Destination directory (batch mode).")
_BLUR = typer.Option(False, "-blur", "--blur")
_MASK = typer.Option(False, "-mask", "--mask")
_ROI = typer.Option(False, "-roi", "--roi")
_ANALYZE = typer.Option(False, "-analyze", "--analyze")
_LANDMARKS = typer.Option(False, "-landmarks", "--landmarks")


@app.command()
def main(
    image: Path | None = _IMAGE,
    src: Path | None = _SRC,
    dst: Path | None = _DST,
    blur: bool = _BLUR,
    mask: bool = _MASK,
    roi: bool = _ROI,
    analyze: bool = _ANALYZE,
    landmarks: bool = _LANDMARKS,
) -> None:
    flags = {
        "blur": blur,
        "mask": mask,
        "roi": roi,
        "analyze": analyze,
        "landmarks": landmarks,
    }

    if image is not None and src is None:
        if not image.is_file():
            die(f"Not a file: {image}")
        rgb = load_rgb(image)
        outs = all_transforms(rgb)
        hist = color_histogram(rgb)

        # One figure: top row = 6 transforms, bottom = full-width histogram.
        fig = plt.figure(figsize=(16, 9))
        for i, (name, img) in enumerate(outs.items()):
            ax = plt.subplot2grid((2, 6), (0, i), fig=fig)
            # Lock 2-D (binary / grayscale) panels to a fixed 0-255 range so
            # near-uniform masks don't get auto-normalised to all-black.
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.set_title(name, fontsize=10)
            ax.axis("off")
        ax_hist = plt.subplot2grid((2, 6), (1, 0), colspan=6, fig=fig)
        for name, values in hist.items():
            ax_hist.plot(values, label=name)
        ax_hist.set_title("Color histogram")
        ax_hist.set_xlabel("Pixel intensity")
        ax_hist.set_ylabel("Proportion of pixels (%)")
        ax_hist.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Transformations: {image.name}", fontsize=11)
        plt.tight_layout()
        plt.show()
        return

    if src is None or dst is None:
        die("Batch mode requires both -src and -dst.")
    if not src.is_dir():
        die(f"Source is not a directory: {src}")

    chosen = [FLAG_TO_KEY[k] for k, v in flags.items() if v]
    if not chosen:
        chosen = list(FLAG_TO_KEY.values())  # all by default
    dst.mkdir(parents=True, exist_ok=True)
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    images = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes]
    for img_path in track(images, description="Transforming"):
        rgb = load_rgb(img_path)
        outs = all_transforms(rgb)
        for key in chosen:
            out_path = dst / f"{img_path.stem}_{key}{img_path.suffix}"
            Image.fromarray(outs[key]).save(out_path)
    console.print(f"[ok]Wrote {len(images) * len(chosen)} files into {dst}[/ok]")


if __name__ == "__main__":
    app()
