"""Tests for LeafDataset & discover_classes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from leaffliction.dataset import LeafDataset, discover_classes


def _make_image(path: Path) -> None:
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_discover_classes_flat_layout(tmp_path: Path) -> None:
    """images/ClassA/*.JPG, images/ClassB/*.JPG"""
    for cls in ("Apple_healthy", "Apple_rust"):
        d = tmp_path / cls
        d.mkdir()
        _make_image(d / "image (1).JPG")
        _make_image(d / "image (2).JPG")
    classes = discover_classes(tmp_path)
    assert sorted(classes.keys()) == ["Apple_healthy", "Apple_rust"]
    assert all(len(paths) == 2 for paths in classes.values())


def test_discover_classes_nested_layout(tmp_path: Path) -> None:
    """images/Apple/apple_healthy/*.JPG, images/Apple/apple_rust/*.JPG"""
    for sub in ("apple_healthy", "apple_rust"):
        d = tmp_path / "Apple" / sub
        d.mkdir(parents=True)
        _make_image(d / "image (1).JPG")
    classes = discover_classes(tmp_path)
    assert sorted(classes.keys()) == ["apple_healthy", "apple_rust"]


def test_discover_classes_single_class_dir(tmp_path: Path) -> None:
    """When passed a single leaf directory of images, treat folder as one class."""
    _make_image(tmp_path / "image (1).JPG")
    _make_image(tmp_path / "image (2).JPG")
    classes = discover_classes(tmp_path)
    assert list(classes.keys()) == [tmp_path.name]
    assert len(classes[tmp_path.name]) == 2


def test_discover_classes_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No images found"):
        discover_classes(tmp_path)


def test_leaf_dataset_builds_index(tmp_path: Path) -> None:
    """LeafDataset wires class index from discover_classes output."""
    for cls in ("Apple_healthy", "Apple_rust"):
        d = tmp_path / cls
        d.mkdir()
        _make_image(d / "image (1).JPG")
    ds = LeafDataset(tmp_path)
    assert len(ds) == 2
    assert ds.classes == ["Apple_healthy", "Apple_rust"]
    assert ds.class_to_idx == {"Apple_healthy": 0, "Apple_rust": 1}


def test_dataset_getitem_returns_tensor(tmp_dataset: Path) -> None:
    import torch

    ds = LeafDataset(tmp_dataset)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min() <= img.max() <= 1.0
    assert label in {0, 1}


def test_dataset_len_matches_total_images(tmp_dataset: Path) -> None:
    ds = LeafDataset(tmp_dataset)
    assert len(ds) == 6  # 2 classes × 3 images


def test_scratch_cnn_forward_shape() -> None:
    import torch

    from leaffliction.models.scratch_cnn import ScratchCNN

    model = ScratchCNN(num_classes=8)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 8)
