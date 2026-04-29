"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    """Build a tiny synthetic dataset with two classes, three images each."""
    for cls in ("ClassA", "ClassB"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"image ({i + 1}).JPG")
    return tmp_path
