"""Smart folder-walker that turns any leaf-image directory into class -> [paths]."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F  # noqa: N812

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if _is_image(p))


def discover_classes(root: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} from an arbitrary directory layout.

    Handles three layouts:
    1. Single leaf directory (root contains images directly) -> 1 class named after root
    2. Flat: root/<class>/*.jpg
    3. Nested: root/<group>/<class>/*.jpg (group folder is ignored, class is the leaf)
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    direct_images = _images_in(root)
    if direct_images:
        return {root.name: direct_images}

    classes: dict[str, list[Path]] = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        child_images = _images_in(child)
        if child_images:
            classes[child.name] = child_images
            continue
        for grand in sorted(p for p in child.iterdir() if p.is_dir()):
            grand_images = _images_in(grand)
            if grand_images:
                classes[grand.name] = grand_images
    if not classes:
        raise ValueError(f"No images found under {root}")
    return classes


class LeafDataset(Dataset):
    """torch Dataset wrapping discover_classes output. __getitem__ implemented in Task 4."""

    def __init__(self, root: Path, transform=None) -> None:
        self.classes_map = discover_classes(Path(root))
        self.classes = sorted(self.classes_map.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = [
            (path, self.class_to_idx[cls])
            for cls, paths in self.classes_map.items()
            for path in paths
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = F.to_image(img)
        tensor = F.to_dtype(tensor, dtype=torch.float32, scale=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label
