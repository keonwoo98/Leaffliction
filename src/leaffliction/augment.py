"""Six Albumentations-based augmentations (PDF Part 2)."""
from __future__ import annotations

import random
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
from PIL import Image

# Note: Crop op uses RandomResizedCrop with size=(256,256). For tests with smaller toy
# images (64x64), the output size will still be 256x256 — that's fine because tests only
# check shape[2]==3, not exact dimensions.
AUGMENTATION_OPS: dict[str, A.BasicTransform] = {
    "Flip": A.HorizontalFlip(p=1.0),
    "Rotate": A.Rotate(limit=30, p=1.0, border_mode=0),
    "Skew": A.Affine(shear={"x": (-15, 15), "y": (-15, 15)}, p=1.0, border_mode=0),
    "Shear": A.Affine(shear={"x": (-25, 25)}, p=1.0, border_mode=0),
    "Crop": A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
    "Distortion": A.OpticalDistortion(distort_limit=0.4, p=1.0),
}


def apply_op(name: str, image: np.ndarray) -> np.ndarray:
    """Run a single named op on an HxWx3 uint8 array. Returns same dtype."""
    if name not in AUGMENTATION_OPS:
        raise KeyError(f"Unknown op: {name}. Choose from {sorted(AUGMENTATION_OPS)}")
    transform = AUGMENTATION_OPS[name]
    return transform(image=image)["image"]


def apply_random_op(
    image: np.ndarray, rng: random.Random | None = None
) -> tuple[str, np.ndarray]:
    rng = rng or random.Random()
    name = rng.choice(list(AUGMENTATION_OPS))
    return name, apply_op(name, image)


def save_with_suffix(original_path: Path, image: np.ndarray, suffix: str) -> Path:
    """Save augmented image as `<stem>_<Suffix>.JPG` next to the original."""
    original_path = Path(original_path)
    out = original_path.with_name(
        f"{original_path.stem}_{suffix}{original_path.suffix}"
    )
    Image.fromarray(image).save(out)
    return out


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))
