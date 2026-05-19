"""Six Albumentations-based augmentations (PDF Part 2)."""

from __future__ import annotations

import random
import shutil
import zipfile
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


def apply_random_op(image: np.ndarray, rng: random.Random | None = None) -> tuple[str, np.ndarray]:
    rng = rng or random.Random()
    name = rng.choice(list(AUGMENTATION_OPS))
    return name, apply_op(name, image)


def save_with_suffix(original_path: Path, image: np.ndarray, suffix: str) -> Path:
    """Save augmented image as `<stem>_<Suffix>.JPG` next to the original."""
    original_path = Path(original_path)
    out = original_path.with_name(f"{original_path.stem}_{suffix}{original_path.suffix}")
    Image.fromarray(image).save(out)
    return out


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def zip_directory(directory: Path) -> Path:
    """Create <directory>.zip next to the directory itself.

    The resulting archive uses the directory's basename as the top-level entry
    so unzipping produces the same folder structure.
    """
    directory = Path(directory)
    zip_path = directory.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in directory.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(directory.parent))
    return zip_path


def balance_directory(
    src_root: Path,
    dst_root: Path,
    target_count: int | None = None,
    seed: int = 42,
    make_zip: bool = True,
) -> dict[str, int]:
    """Copy originals + generate augmentations so every class reaches target_count.

    If target_count is None, uses the largest class size.
    By default also creates <dst_root>.zip alongside the folder (PDF Chapter V
    requires this archive for the defense signature check).
    Returns {class_name: final_count}.
    """
    from leaffliction.dataset import discover_classes  # avoid circular at import

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    rng = random.Random(seed)

    classes = discover_classes(src_root)
    target = target_count if target_count is not None else max(len(p) for p in classes.values())

    summary: dict[str, int] = {}
    for cls, paths in classes.items():
        cls_dst = dst_root / cls
        cls_dst.mkdir(parents=True, exist_ok=True)
        # 1) copy originals (cap at target if class already exceeds it)
        originals_to_copy = paths[:target]
        for src in originals_to_copy:
            shutil.copy2(src, cls_dst / src.name)
        produced = len(originals_to_copy)
        # 2) augment until target reached
        bump = 0
        while produced < target:
            base = rng.choice(paths)
            img = load_image(base)
            op_name, aug = apply_random_op(img, rng)
            out = cls_dst / f"{base.stem}_{op_name}_{bump}{base.suffix}"
            Image.fromarray(aug).save(out)
            produced += 1
            bump += 1
        summary[cls] = produced

    if make_zip:
        zip_directory(dst_root)

    return summary
