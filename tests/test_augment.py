"""Tests for the 6 augmentation ops + filename suffix convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from leaffliction.augment import AUGMENTATION_OPS, apply_op, save_with_suffix

EXPECTED_OPS = {"Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"}


def _toy_image(size: int = 64) -> np.ndarray:
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def test_six_ops_present() -> None:
    assert set(AUGMENTATION_OPS) == EXPECTED_OPS


@pytest.mark.parametrize("op_name", sorted(EXPECTED_OPS))
def test_each_op_returns_uint8_3channel(op_name: str) -> None:
    img = _toy_image()
    out = apply_op(op_name, img)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_save_with_suffix(tmp_path: Path) -> None:
    img = _toy_image()
    src = tmp_path / "image (1).JPG"
    Image.fromarray(img).save(src)
    out = save_with_suffix(src, img, "Flip")
    assert out.name == "image (1)_Flip.JPG"
    assert out.exists()


def test_balance_directory_creates_target_count(tmp_dataset: Path, tmp_path: Path) -> None:
    from leaffliction.augment import balance_directory

    out_root = tmp_path / "augmented"
    summary = balance_directory(tmp_dataset, out_root, target_count=5, seed=0)
    # tmp_dataset has 2 classes × 3 images. target=5 → 5 each
    for cls in ("ClassA", "ClassB"):
        files = list((out_root / cls).iterdir())
        assert len(files) == 5
    assert summary["ClassA"] == 5
    assert summary["ClassB"] == 5
