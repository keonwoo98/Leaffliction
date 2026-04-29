# Leaffliction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the five 42-Leaffliction programs (Distribution / Augmentation / Transformation / train / predict) using PyTorch 2.x + torchvision, with two parallel models (ScratchCNN + Transfer EfficientNet-B0), Albumentations, plantCV, signed artifacts, and a defense-proof testing/lint pipeline.

**Architecture:** Five thin entrypoint scripts at the repo root call into a `src/leaffliction/` Python package. A smart dataset walker (`LeafDataset`) is shared across all five programs. Training produces two `.pt` files plus metadata, all packaged into `trained_models.zip` together with `augmented_directory.zip`; their SHA1 hashes are written to `signature.txt` automatically.

**Tech Stack:** Python 3.12, PyTorch 2.x, torchvision, Albumentations, plantCV, typer, rich, matplotlib, seaborn, scikit-learn, pytest, ruff, flake8, uv.

**Reference:** [docs/superpowers/specs/2026-04-28-leaffliction-design.md](../specs/2026-04-28-leaffliction-design.md)

---

## File Structure

### Files to create

| Path | Responsibility |
|------|---------------|
| `pyproject.toml` | uv project config, deps, ruff/flake8/pytest settings |
| `.python-version` | pin Python 3.12 |
| `.gitignore` | block dataset/model artifacts |
| `.pre-commit-config.yaml` | ruff + dataset-block guard |
| `Makefile` | one-line shortcuts (lint, format, test, train, verify) |
| `README.md` | usage, defense checklist |
| `src/leaffliction/__init__.py` | package marker, version |
| `src/leaffliction/dataset.py` | `LeafDataset`, `discover_classes` |
| `src/leaffliction/signature.py` | `compute_sha1`, `write_signature`, `verify_signature` |
| `src/leaffliction/viz.py` | `pie_and_bar`, `grid`, `learning_curves`, `confusion_matrix_plot` |
| `src/leaffliction/augment.py` | 6 Albumentations ops + `balance_directory` |
| `src/leaffliction/transform.py` | 6 plantCV transforms + 9-channel histogram |
| `src/leaffliction/models/__init__.py` | exports |
| `src/leaffliction/models/scratch_cnn.py` | `ScratchCNN(nn.Module)` |
| `src/leaffliction/models/transfer.py` | `TransferModel` (EfficientNet-B0) |
| `src/leaffliction/trainer.py` | `Trainer` (training loop, early stop, scheduler) |
| `src/leaffliction/predictor.py` | `Predictor` (load zip, run inference, render figure) |
| `src/leaffliction/seed.py` | `set_seed` |
| `src/leaffliction/cli.py` | shared typer/rich helpers |
| `Distribution.py` | Part 1 entrypoint |
| `Augmentation.py` | Part 2 entrypoint |
| `Transformation.py` | Part 3 entrypoint |
| `train.py` | Part 4 entrypoint |
| `predict.py` | Part 4 entrypoint |
| `tests/__init__.py` | empty |
| `tests/conftest.py` | shared fixtures (tmp dataset factory) |
| `tests/test_dataset.py` | discover_classes + Dataset semantics |
| `tests/test_signature.py` | sha1 vs `shasum`, verify mismatch |
| `tests/test_augment.py` | 6 ops shape/dtype + filename suffix |
| `tests/test_smoke.py` | 5 entrypoints `--help` exit 0 |
| `scripts/verify.sh` | compare signature.txt to actual zip hashes |
| `scripts/check_no_dataset.sh` | pre-commit guard |

### Boundaries

- **Pure data**: `dataset.py` (no PyTorch imports beyond `torch.utils.data.Dataset` interface)
- **Pure CV**: `augment.py`, `transform.py` (no PyTorch dependence)
- **Pure ML**: `models/`, `trainer.py`, `predictor.py`
- **Glue**: `cli.py`, `seed.py`, `viz.py`, `signature.py`
- **Entry points**: 5 root scripts, each ~30-50 lines, only call into the package

---

## Phase 0 — Foundation

### Task 1: Initialize uv project & toolchain

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `Makefile`

- [ ] **Step 1: Verify uv is installed**

```bash
uv --version
# If missing on macOS: curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: brew install uv
```
Expected: prints version like `uv 0.5.x`.

- [ ] **Step 2: Create `.python-version`**

```
3.12
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "leaffliction"
version = "1.0.0"
description = "42 Leaffliction — leaf disease image classification"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "albumentations>=1.4",
    "plantcv>=4.0",
    "pillow",
    "opencv-python",
    "matplotlib",
    "seaborn",
    "numpy",
    "scikit-learn",
    "typer>=0.12",
    "rich",
]

[dependency-groups]
dev = ["ruff", "flake8", "pytest", "pytest-cov", "pre-commit"]

[tool.hatch.build.targets.wheel]
packages = ["src/leaffliction"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 99
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM"]
ignore = ["E501"]  # line length handled by formatter

[tool.flake8]
max-line-length = 99
extend-ignore = ["E203", "W503"]
exclude = [".venv", "build", "dist"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
pythonpath = ["src"]
```

- [ ] **Step 4: Create `.gitignore`**

```gitignore
# Datasets and models — never commit
images/
augmented_directory/
artifacts/
*.zip
*.pt
*.pth

# Python
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
dist/

# OS / editors
.DS_Store
.vscode/
.idea/
```

- [ ] **Step 5: Create `Makefile`**

```make
.PHONY: setup lint format test smoke train verify clean

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run flake8 src tests *.py

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest --cov=src/leaffliction

smoke:
	uv run pytest tests/test_smoke.py -v

train:
	uv run python train.py images/

verify:
	bash scripts/verify.sh

clean:
	rm -rf .venv .pytest_cache .ruff_cache build dist *.egg-info
```

- [ ] **Step 6: Run `uv sync`**

```bash
uv sync
```
Expected: creates `.venv/`, installs all deps, writes `uv.lock`. Takes ~30s.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version .gitignore Makefile uv.lock
git commit -m "chore: bootstrap uv project + toolchain config"
```

---

### Task 2: Package skeleton + smoke test infrastructure

**Files:**
- Create: `src/leaffliction/__init__.py`
- Create: `src/leaffliction/seed.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

`tests/test_smoke.py`:
```python
"""Smoke tests: each entrypoint runs `--help` without crashing."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ENTRYPOINTS = ["Distribution.py", "Augmentation.py", "Transformation.py", "train.py", "predict.py"]


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_entrypoint_help_exits_zero(script: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / script
    if not path.exists():
        pytest.skip(f"{script} not yet implemented")
    result = subprocess.run(
        ["uv", "run", "python", str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{script} --help failed: {result.stderr}"
```

- [ ] **Step 2: Run smoke test (expect skips, not failures)**

```bash
uv run pytest tests/test_smoke.py -v
```
Expected: 5 SKIPPED (no entrypoints yet), no FAIL.

- [ ] **Step 3: Create package files**

`src/leaffliction/__init__.py`:
```python
"""Leaffliction — leaf disease image classification (42 project)."""

__version__ = "1.0.0"
```

`src/leaffliction/seed.py`:
```python
"""Centralized random seeding for reproducibility."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG that may affect training/eval results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

`tests/__init__.py`:
```python
```

`tests/conftest.py`:
```python
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
```

- [ ] **Step 4: Run smoke test again**

```bash
uv run pytest tests/test_smoke.py -v
```
Expected: still 5 SKIPPED — entrypoints still missing.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction tests
git commit -m "feat(core): package skeleton + smoke test scaffolding"
```

---

## Phase 1 — Core Library (TDD)

### Task 3: `LeafDataset.discover_classes` (smart folder walker)

**Files:**
- Create: `src/leaffliction/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

```bash
uv run pytest tests/test_dataset.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'leaffliction.dataset'`.

- [ ] **Step 3: Implement `discover_classes`**

`src/leaffliction/dataset.py`:
```python
"""Smart folder-walker that turns any leaf-image directory into class -> [paths]."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if _is_image(p))


def discover_classes(root: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image_path, ...]} from an arbitrary directory layout.

    Handles three layouts:
    1. Single leaf directory (root contains images directly) → 1 class named after root
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
    """torch Dataset wrapping discover_classes output. Implemented in Task 4."""

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

    def __getitem__(self, idx: int):  # implementation in Task 4
        raise NotImplementedError("Implemented in Task 4")
```

- [ ] **Step 4: Run test, expect PASS**

```bash
uv run pytest tests/test_dataset.py::test_discover_classes_flat_layout -v
uv run pytest tests/test_dataset.py::test_discover_classes_nested_layout -v
uv run pytest tests/test_dataset.py::test_discover_classes_single_class_dir -v
uv run pytest tests/test_dataset.py::test_discover_classes_raises_when_empty -v
```
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): smart class discovery for any layout"
```

---

### Task 4: `LeafDataset.__getitem__` (PyTorch Dataset interface)

**Files:**
- Modify: `src/leaffliction/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_dataset.py`:
```python
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
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/test_dataset.py::test_dataset_getitem_returns_tensor -v
```
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `__getitem__`**

Replace the body of `__getitem__` in `src/leaffliction/dataset.py`:
```python
    def __getitem__(self, idx: int):
        from PIL import Image
        from torchvision.transforms.v2 import functional as F

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = F.to_image(img)
        tensor = F.to_dtype(tensor, dtype=torch.float32, scale=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label
```

Add `import torch` at the top of the file.

- [ ] **Step 4: Run, expect PASS**

```bash
uv run pytest tests/test_dataset.py -v
```
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): implement __getitem__ returning float tensor"
```

---

### Task 5: `signature.py` (SHA1 + write/verify)

**Files:**
- Create: `src/leaffliction/signature.py`
- Create: `tests/test_signature.py`

- [ ] **Step 1: Write failing test**

`tests/test_signature.py`:
```python
"""Tests for SHA1 signing and signature.txt round-trip."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from leaffliction.signature import compute_sha1, verify_signature, write_signature


def test_compute_sha1_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hello leaffliction"
    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert compute_sha1(f) == hashlib.sha1(payload).hexdigest()


def test_compute_sha1_matches_shasum(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"sha-cmd-cross-check")
    cmd = "shasum" if subprocess.run(["which", "shasum"], capture_output=True).returncode == 0 else "sha1sum"
    out = subprocess.check_output([cmd, str(f)], text=True).split()[0]
    assert compute_sha1(f) == out


def test_write_and_verify_roundtrip(tmp_path: Path) -> None:
    a = tmp_path / "a.zip"
    a.write_bytes(b"AAAA")
    b = tmp_path / "b.zip"
    b.write_bytes(b"BBBB")
    sig_path = tmp_path / "signature.txt"
    write_signature([a, b], sig_path)
    assert sig_path.exists()
    assert verify_signature(sig_path) is True


def test_verify_detects_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.zip"
    a.write_bytes(b"original")
    sig_path = tmp_path / "signature.txt"
    write_signature([a], sig_path)
    a.write_bytes(b"tampered")
    assert verify_signature(sig_path) is False


def test_verify_missing_file_returns_false(tmp_path: Path) -> None:
    sig_path = tmp_path / "signature.txt"
    sig_path.write_text("0000000000000000000000000000000000000000  ghost.zip\n")
    assert verify_signature(sig_path) is False
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/test_signature.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement signature module**

`src/leaffliction/signature.py`:
```python
"""SHA1 signing & verification for dataset/model zip artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 64 * 1024


def compute_sha1(path: Path) -> str:
    """Stream the file through sha1; matches `shasum`/`sha1sum`."""
    h = hashlib.sha1()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_signature(paths: list[Path], out: Path) -> None:
    """Write `<sha1>  <basename>\\n` per file (compatible with `shasum -c`)."""
    lines: list[str] = []
    for p in paths:
        p = Path(p)
        digest = compute_sha1(p)
        lines.append(f"{digest}  {p.name}")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_signature(sig_path: Path) -> bool:
    """Return True iff every recorded hash matches a sibling file of `sig_path`."""
    sig_path = Path(sig_path)
    if not sig_path.exists():
        return False
    base = sig_path.parent
    for raw in sig_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        target = base / name.strip()
        if not target.exists():
            return False
        if compute_sha1(target) != digest.strip():
            return False
    return True
```

- [ ] **Step 4: Run, expect PASS**

```bash
uv run pytest tests/test_signature.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/signature.py tests/test_signature.py
git commit -m "feat(signature): SHA1 compute/write/verify for zip artifacts"
```

---

### Task 6: `viz.py` (charts)

**Files:**
- Create: `src/leaffliction/viz.py`

- [ ] **Step 1: Write the module**

`src/leaffliction/viz.py`:
```python
"""Matplotlib chart helpers used across all 5 entrypoints."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PALETTE = sns.color_palette("Set2")


def pie_and_bar(class_counts: dict[str, int], title: str, save: Path | None = None) -> None:
    """Render a side-by-side pie+bar of class distribution (Part 1)."""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(counts, labels=classes, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[0].set_title(f"{title} class distribution")

    axes[1].bar(classes, counts, color=colors)
    axes[1].set_xticklabels(classes, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def grid(images: list[tuple[str, np.ndarray]], save: Path | None = None) -> None:
    """Render N labelled images in a single row (used by Augmentation/Transformation)."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, images, strict=True):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def learning_curves(history: dict[str, list[float]], save: Path) -> None:
    """Two-axis loss/accuracy plot for train.py."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix_plot(cm: np.ndarray, classes: list[str], save: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 2: Lint check (no test, just visual module)**

```bash
uv run ruff check src/leaffliction/viz.py
uv run flake8 src/leaffliction/viz.py
```
Expected: zero issues.

- [ ] **Step 3: Commit**

```bash
git add src/leaffliction/viz.py
git commit -m "feat(viz): pie/bar/grid/learning-curve/confusion-matrix helpers"
```

---

### Task 7: `augment.py` — 6 Albumentations ops

**Files:**
- Create: `src/leaffliction/augment.py`
- Create: `tests/test_augment.py`

- [ ] **Step 1: Write failing test**

`tests/test_augment.py`:
```python
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
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/test_augment.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement augment module**

`src/leaffliction/augment.py`:
```python
"""Six Albumentations-based augmentations (PDF Part 2)."""
from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

AUGMENTATION_OPS: dict[str, A.BasicTransform] = {
    "Flip": A.HorizontalFlip(p=1.0),
    "Rotate": A.Rotate(limit=30, p=1.0, border_mode=0),
    "Skew": A.Affine(shear={"x": (-15, 15), "y": (-15, 15)}, p=1.0, mode=0),
    "Shear": A.Affine(shear={"x": (-25, 25)}, p=1.0, mode=0),
    "Crop": A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
    "Distortion": A.OpticalDistortion(distort_limit=0.4, shift_limit=0.1, p=1.0, border_mode=0),
}


def apply_op(name: str, image: np.ndarray) -> np.ndarray:
    """Run a single named op on an HxWx3 uint8 array. Returns same shape/dtype."""
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
```

- [ ] **Step 4: Run, expect PASS**

```bash
uv run pytest tests/test_augment.py -v
```
Expected: 8 PASS (1 + 6 parametric + 1).

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/augment.py tests/test_augment.py
git commit -m "feat(augment): 6 Albumentations ops + filename-suffix saver"
```

---

### Task 8: `augment.balance_directory` (batch mode)

**Files:**
- Modify: `src/leaffliction/augment.py`
- Modify: `tests/test_augment.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_augment.py`:
```python
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
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/test_augment.py::test_balance_directory_creates_target_count -v
```
Expected: FAIL with ImportError.

- [ ] **Step 3: Add `balance_directory`**

Append to `src/leaffliction/augment.py`:
```python
import shutil


def balance_directory(
    src_root: Path,
    dst_root: Path,
    target_count: int | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Copy originals + generate augmentations so every class reaches `target_count`.

    If `target_count` is None, uses the largest class size.
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
        # 1) copy originals
        for src in paths:
            shutil.copy2(src, cls_dst / src.name)
        # 2) augment until target reached
        produced = len(paths)
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
    return summary
```

- [ ] **Step 4: Run, expect PASS**

```bash
uv run pytest tests/test_augment.py -v
```
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/augment.py tests/test_augment.py
git commit -m "feat(augment): balance_directory copies + augments to target count"
```

---

### Task 9: `transform.py` (plantCV wrappers)

**Files:**
- Create: `src/leaffliction/transform.py`

- [ ] **Step 1: Write the module**

`src/leaffliction/transform.py`:
```python
"""plantCV transformations + 9-channel color histogram (PDF Part 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from plantcv import plantcv as pcv

TRANSFORM_NAMES = ("Original", "GaussianBlur", "Mask", "RoiObjects", "AnalyzeObject", "Pseudolandmarks")


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def gaussian_blur(rgb: np.ndarray) -> np.ndarray:
    s = pcv.rgb2gray_hsv(rgb_img=rgb, channel="s")
    binary = pcv.threshold.binary(gray_img=s, threshold=85, object_type="light")
    return pcv.gaussian_blur(img=binary, ksize=(5, 5))


def mask(rgb: np.ndarray) -> np.ndarray:
    blurred = gaussian_blur(rgb)
    cleaned = pcv.fill(bin_img=blurred, size=200)
    return pcv.apply_mask(img=rgb, mask=cleaned, mask_color="white")


def roi_objects(rgb: np.ndarray) -> np.ndarray:
    """Return RGB with leaf contour outlined in green + ROI box in blue."""
    pcv.params.debug = None
    blurred = gaussian_blur(rgb)
    cleaned = pcv.fill(bin_img=blurred, size=200)
    out = rgb.copy()
    contours, _ = __import__("cv2").findContours(cleaned, 0, 2)
    __import__("cv2").drawContours(out, contours, -1, (0, 255, 0), 2)
    h, w = rgb.shape[:2]
    __import__("cv2").rectangle(out, (5, 5), (w - 5, h - 5), (0, 0, 255), 2)
    return out


def analyze_object(rgb: np.ndarray) -> np.ndarray:
    """Run pcv.analyze.size and return its visualization."""
    blurred = gaussian_blur(rgb)
    cleaned = pcv.fill(bin_img=blurred, size=200)
    labeled, _ = pcv.create_labels(mask=cleaned, rois=None, roi_type="partial")
    pcv.analyze.size(img=rgb, labeled_mask=labeled, n_labels=1)
    return pcv.outputs.images[-1] if pcv.outputs.images else rgb


def pseudolandmarks(rgb: np.ndarray) -> np.ndarray:
    blurred = gaussian_blur(rgb)
    cleaned = pcv.fill(bin_img=blurred, size=200)
    out = rgb.copy()
    top, bottom, center = pcv.homology.x_axis_pseudolandmarks(img=rgb, mask=cleaned, label="leaf")
    cv2 = __import__("cv2")
    for color, points in (((0, 0, 255), top), ((255, 165, 0), center), ((255, 0, 255), bottom)):
        for pt in np.atleast_2d(points).reshape(-1, 2).astype(int):
            cv2.circle(out, tuple(pt), 3, color, -1)
    return out


def color_histogram(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return {channel_name: 256-bin histogram (% of pixels)} for 9 channels."""
    cv2 = __import__("cv2")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    channels = {
        "blue": rgb[..., 2],
        "green": rgb[..., 1],
        "red": rgb[..., 0],
        "hue": hsv[..., 0],
        "saturation": hsv[..., 1],
        "value": hsv[..., 2],
        "lightness": lab[..., 0],
        "green-magenta": lab[..., 1],
        "blue-yellow": lab[..., 2],
    }
    out: dict[str, np.ndarray] = {}
    total = rgb.shape[0] * rgb.shape[1]
    for name, ch in channels.items():
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        out[name] = 100.0 * hist / total
    return out


def all_transforms(rgb: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Original": rgb,
        "GaussianBlur": gaussian_blur(rgb),
        "Mask": mask(rgb),
        "RoiObjects": roi_objects(rgb),
        "AnalyzeObject": analyze_object(rgb),
        "Pseudolandmarks": pseudolandmarks(rgb),
    }
```

- [ ] **Step 2: Lint**

```bash
uv run ruff check src/leaffliction/transform.py
uv run flake8 src/leaffliction/transform.py
```
Expected: zero issues.

- [ ] **Step 3: Commit**

```bash
git add src/leaffliction/transform.py
git commit -m "feat(transform): 6 plantCV transforms + 9-channel histogram"
```

---

## Phase 2 — Models

### Task 10: `ScratchCNN`

**Files:**
- Create: `src/leaffliction/models/__init__.py`
- Create: `src/leaffliction/models/scratch_cnn.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_dataset.py`:
```python
def test_scratch_cnn_forward_shape() -> None:
    import torch

    from leaffliction.models.scratch_cnn import ScratchCNN

    model = ScratchCNN(num_classes=8)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 8)
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/test_dataset.py::test_scratch_cnn_forward_shape -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/leaffliction/models/__init__.py`:
```python
from leaffliction.models.scratch_cnn import ScratchCNN
from leaffliction.models.transfer import TransferModel

__all__ = ["ScratchCNN", "TransferModel"]
```

`src/leaffliction/models/scratch_cnn.py`:
```python
"""Hand-designed CNN — every layer is the author's. Defense-friendly."""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
```

- [ ] **Step 4: Run test, expect PASS**

(Note: also need stub `transfer.py` so package imports work. Quick stub now, real impl in Task 11.)

`src/leaffliction/models/transfer.py`:
```python
"""Stub — real implementation in Task 11."""
from __future__ import annotations

from torch import nn


class TransferModel(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x):  # pragma: no cover
        raise NotImplementedError("Implemented in Task 11")
```

```bash
uv run pytest tests/test_dataset.py::test_scratch_cnn_forward_shape -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/models tests/test_dataset.py
git commit -m "feat(models): ScratchCNN with 4 conv blocks + GAP head"
```

---

### Task 11: `TransferModel` (EfficientNet-B0)

**Files:**
- Modify: `src/leaffliction/models/transfer.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_dataset.py`:
```python
def test_transfer_forward_shape_and_freezing() -> None:
    import torch

    from leaffliction.models.transfer import TransferModel

    model = TransferModel(num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 8)
    # By default the backbone is frozen
    backbone_grads = [p.requires_grad for n, p in model.named_parameters() if "classifier" not in n]
    assert not any(backbone_grads)
    # unfreeze() flips them back on
    model.unfreeze()
    backbone_grads = [p.requires_grad for n, p in model.named_parameters() if "classifier" not in n]
    assert all(backbone_grads)
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/test_dataset.py::test_transfer_forward_shape_and_freezing -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Replace `src/leaffliction/models/transfer.py`:
```python
"""EfficientNet-B0 transfer learning with two-stage fine-tuning support."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class TransferModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone
        self.freeze()

    def freeze(self) -> None:
        for name, p in self.model.named_parameters():
            p.requires_grad = "classifier" in name

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

- [ ] **Step 4: Run test, expect PASS**

```bash
uv run pytest tests/test_dataset.py::test_transfer_forward_shape_and_freezing -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/leaffliction/models/transfer.py tests/test_dataset.py
git commit -m "feat(models): TransferModel (EfficientNet-B0) with freeze/unfreeze"
```

---

## Phase 3 — Training & Inference

### Task 12: `Trainer` class

**Files:**
- Create: `src/leaffliction/trainer.py`

- [ ] **Step 1: Implement** (no unit test — covered by integration via train.py later)

`src/leaffliction/trainer.py`:
```python
"""Shared training loop with early stopping + LR plateau scheduler."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-3
    unfreeze_after: int | None = None  # for transfer model: epoch at which to unfreeze


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=lambda: {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    })
    best_val_acc: float = 0.0
    best_epoch: int = 0
    state_dict: dict[str, Any] | None = None


def _epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
           optimizer: optim.Optimizer | None, device: torch.device) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += float(loss) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    device = device_auto()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    result = TrainResult()
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, config.epochs + 1):
        if config.unfreeze_after is not None and epoch == config.unfreeze_after + 1 and hasattr(model, "unfreeze"):
            model.unfreeze()
            optimizer = optim.Adam(model.parameters(), lr=config.lr * 0.1, weight_decay=config.weight_decay)

        train_loss, train_acc = _epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch(model, val_loader, criterion, None, device)

        result.history["train_loss"].append(train_loss)
        result.history["val_loss"].append(val_loss)
        result.history["train_acc"].append(train_acc)
        result.history["val_acc"].append(val_acc)

        scheduler.step(val_acc)
        if val_acc > result.best_val_acc + config.min_delta:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch, {
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            })

        if epochs_no_improve >= config.patience:
            break

    result.state_dict = best_state
    return result


def save_state(state: dict[str, Any], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
```

- [ ] **Step 2: Lint**

```bash
uv run ruff check src/leaffliction/trainer.py
uv run flake8 src/leaffliction/trainer.py
```
Expected: zero issues.

- [ ] **Step 3: Commit**

```bash
git add src/leaffliction/trainer.py
git commit -m "feat(trainer): training loop with early stop + LR scheduler + unfreeze hook"
```

---

### Task 13: `Predictor` class

**Files:**
- Create: `src/leaffliction/predictor.py`

- [ ] **Step 1: Implement**

`src/leaffliction/predictor.py`:
```python
"""Inference: loads a zip artifact, classifies one image, renders a figure."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F

from leaffliction.models import ScratchCNN, TransferModel
from leaffliction.transform import mask as mask_transform

ImageNetMean = [0.485, 0.456, 0.406]
ImageNetStd = [0.229, 0.224, 0.225]


def _build_model(name: str, num_classes: int) -> torch.nn.Module:
    if name == "scratch":
        return ScratchCNN(num_classes=num_classes)
    if name == "transfer":
        return TransferModel(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model name: {name}")


def _preprocess(rgb: np.ndarray, size: int = 256) -> torch.Tensor:
    img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    t = F.to_image(img)
    t = F.to_dtype(t, dtype=torch.float32, scale=True)
    t = F.normalize(t, mean=ImageNetMean, std=ImageNetStd)
    return t.unsqueeze(0)


def predict(image_path: Path, zip_path: Path, prefer: str = "transfer") -> dict:
    """Return {'class': str, 'confidence': float, 'rgb': np.ndarray, 'transformed': np.ndarray}."""
    image_path = Path(image_path)
    zip_path = Path(zip_path)

    extract_dir = zip_path.parent / f".{zip_path.stem}_unpacked"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    metadata = json.loads((extract_dir / "metadata.json").read_text())
    classes: list[str] = metadata["classes"]
    weight_file = extract_dir / f"model_{prefer}.pt"
    if not weight_file.exists():
        # fallback if requested model wasn't trained
        for alt in ("transfer", "scratch"):
            weight_file = extract_dir / f"model_{alt}.pt"
            if weight_file.exists():
                prefer = alt
                break

    model = _build_model(prefer, num_classes=len(classes))
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()

    rgb = np.array(Image.open(image_path).convert("RGB"))
    tensor = _preprocess(rgb, size=metadata.get("image_size", 256))
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
    return {
        "class": classes[idx],
        "confidence": float(probs[idx]),
        "rgb": rgb,
        "transformed": mask_transform(rgb),
    }


def render(result: dict, save: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.05})
    axes[0].imshow(result["rgb"])
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(result["transformed"])
    axes[1].set_title("Mask transform")
    axes[1].axis("off")

    fig.suptitle(
        f"=== DL classification ===\nClass predicted : {result['class']} ({result['confidence']:.1%})",
        fontsize=14,
        color="#39c46a",
    )
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
```

- [ ] **Step 2: Lint**

```bash
uv run ruff check src/leaffliction/predictor.py
uv run flake8 src/leaffliction/predictor.py
```
Expected: zero issues.

- [ ] **Step 3: Commit**

```bash
git add src/leaffliction/predictor.py
git commit -m "feat(predictor): zip-loading inference with original/mask figure"
```

---

### Task 14: `cli.py` (typer helpers)

**Files:**
- Create: `src/leaffliction/cli.py`

- [ ] **Step 1: Implement**

`src/leaffliction/cli.py`:
```python
"""Shared typer/rich helpers for the 5 entrypoints."""
from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

THEME = Theme({"info": "cyan", "warn": "yellow", "err": "bold red", "ok": "bold green"})
console = Console(theme=THEME)


def die(msg: str, code: int = 1) -> "None":
    console.print(f"[err]error:[/err] {msg}")
    sys.exit(code)
```

- [ ] **Step 2: Commit**

```bash
git add src/leaffliction/cli.py
git commit -m "feat(cli): rich console helper used by all entrypoints"
```

---

## Phase 4 — Entrypoints

### Task 15: `Distribution.py`

**Files:**
- Create: `Distribution.py`

- [ ] **Step 1: Implement**

`Distribution.py`:
```python
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
    directory: Path = typer.Argument(..., exists=True, help="Path to dataset root."),
    save: Path | None = typer.Option(None, "--save", help="If set, save chart PNG instead of showing."),
) -> None:
    try:
        classes = discover_classes(directory)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        die(str(exc))
    counts = {name: len(paths) for name, paths in classes.items()}
    title = directory.name or directory.resolve().name
    console.print(f"[info]Found {len(counts)} classes, {sum(counts.values())} images in {directory}[/info]")
    pie_and_bar(counts, title=title, save=save)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Make executable**

```bash
chmod +x Distribution.py
```

- [ ] **Step 3: Verify --help works**

```bash
uv run python Distribution.py --help
```
Expected: typer help output, exit 0.

- [ ] **Step 4: Run smoke**

```bash
uv run pytest tests/test_smoke.py::test_entrypoint_help_exits_zero -v -k Distribution
```
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add Distribution.py
git commit -m "feat(part1): Distribution.py — pie + bar chart of dataset classes"
```

---

### Task 16: `Augmentation.py`

**Files:**
- Create: `Augmentation.py`

- [ ] **Step 1: Implement**

`Augmentation.py`:
```python
#!/usr/bin/env python3
"""Part 2 — augmentation: 6 ops on a single image, or balance a directory."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.progress import track

from leaffliction.augment import (
    AUGMENTATION_OPS,
    apply_op,
    balance_directory,
    load_image,
    save_with_suffix,
)
from leaffliction.cli import console, die
from leaffliction.viz import grid

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    target: Path = typer.Argument(..., exists=True, help="Image file (single mode) or directory (batch mode)."),
    balance: bool = typer.Option(False, "--balance", help="Batch mode: balance classes via augmentation."),
    output: Path = typer.Option(Path("augmented_directory"), "--output", "-o", help="Batch mode output dir."),
    target_count: int | None = typer.Option(None, "--target-count", help="Per-class target count (default: max class)."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    if target.is_file():
        rgb = load_image(target)
        outputs = [("Original", rgb)]
        for name in AUGMENTATION_OPS:
            aug = apply_op(name, rgb)
            save_with_suffix(target, aug, name)
            outputs.append((name, aug))
        console.print(f"[ok]Saved 6 augmentations next to {target.name}[/ok]")
        grid(outputs)
        return

    if not balance:
        die("Directory passed without --balance. Pass --balance to run batch mode.")

    console.print(f"[info]Balancing {target} into {output} ...[/info]")
    summary = balance_directory(target, output, target_count=target_count, seed=seed)
    for cls, n in summary.items():
        console.print(f"  [info]{cls}[/info]: {n} images")
    console.print(f"[ok]Done. Output at {output}[/ok]")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: chmod + verify**

```bash
chmod +x Augmentation.py
uv run python Augmentation.py --help
```
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add Augmentation.py
git commit -m "feat(part2): Augmentation.py — single mode + batch balance mode"
```

---

### Task 17: `Transformation.py`

**Files:**
- Create: `Transformation.py`

- [ ] **Step 1: Implement**

`Transformation.py`:
```python
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
from leaffliction.viz import grid

app = typer.Typer(add_completion=False, help=__doc__, context_settings={"help_option_names": ["-h", "--help"]})

FLAG_TO_KEY = {
    "blur": "GaussianBlur",
    "mask": "Mask",
    "roi": "RoiObjects",
    "analyze": "AnalyzeObject",
    "landmarks": "Pseudolandmarks",
}


@app.command()
def main(
    image: Path | None = typer.Argument(None, help="Single image path (display mode)."),
    src: Path | None = typer.Option(None, "-src", "--src", help="Source directory (batch mode)."),
    dst: Path | None = typer.Option(None, "-dst", "--dst", help="Destination directory (batch mode)."),
    blur: bool = typer.Option(False, "-blur", "--blur"),
    mask: bool = typer.Option(False, "-mask", "--mask"),
    roi: bool = typer.Option(False, "-roi", "--roi"),
    analyze: bool = typer.Option(False, "-analyze", "--analyze"),
    landmarks: bool = typer.Option(False, "-landmarks", "--landmarks"),
) -> None:
    flags = {"blur": blur, "mask": mask, "roi": roi, "analyze": analyze, "landmarks": landmarks}

    if image is not None and src is None:
        if not image.is_file():
            die(f"Not a file: {image}")
        rgb = load_rgb(image)
        outs = all_transforms(rgb)
        grid(list(outs.items()))
        hist = color_histogram(rgb)
        plt.figure(figsize=(10, 5))
        for name, values in hist.items():
            plt.plot(values, label=name)
        plt.title("Color histogram")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Proportion of pixels (%)")
        plt.legend()
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
    images = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for img_path in track(images, description="Transforming"):
        rgb = load_rgb(img_path)
        outs = all_transforms(rgb)
        for key in chosen:
            out_path = dst / f"{img_path.stem}_{key}{img_path.suffix}"
            Image.fromarray(outs[key]).save(out_path)
    console.print(f"[ok]Wrote {len(images) * len(chosen)} files into {dst}[/ok]")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: chmod + verify**

```bash
chmod +x Transformation.py
uv run python Transformation.py -h
```
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add Transformation.py
git commit -m "feat(part3): Transformation.py — single + batch with -src/-dst/flags"
```

---

### Task 18: `train.py`

**Files:**
- Create: `train.py`

- [ ] **Step 1: Implement**

`train.py`:
```python
#!/usr/bin/env python3
"""Part 4 — train ScratchCNN and TransferModel; package artifacts + signature."""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms.v2 import Compose, Normalize, RandomHorizontalFlip, RandomRotation

from leaffliction.cli import console, die
from leaffliction.dataset import LeafDataset
from leaffliction.models import ScratchCNN, TransferModel
from leaffliction.predictor import ImageNetMean, ImageNetStd
from leaffliction.seed import set_seed
from leaffliction.signature import write_signature
from leaffliction.trainer import TrainConfig, save_state, train
from leaffliction.viz import confusion_matrix_plot, learning_curves

app = typer.Typer(add_completion=False, help=__doc__)


def _build_loaders(directory: Path, split: float, batch: int, seed: int):
    """Build DataLoaders.

    Two LeafDataset instances over the same directory are created so that train and val
    can have different transforms (augmentation only on train). discover_classes returns
    sorted output, so the two instances iterate samples in identical order — making the
    integer indices from train_test_split consistent across both.
    """
    train_tf = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        Normalize(mean=ImageNetMean, std=ImageNetStd),
    ])
    val_tf = Compose([Normalize(mean=ImageNetMean, std=ImageNetStd)])

    train_full = LeafDataset(directory, transform=train_tf)
    val_full = LeafDataset(directory, transform=val_tf)
    labels = [lab for _, lab in train_full.samples]

    train_idx, val_idx = train_test_split(
        list(range(len(labels))),
        test_size=1 - split,
        stratify=labels,
        random_state=seed,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    # WeightedRandomSampler over the train subset only
    train_labels = np.array([labels[i] for i in train_idx])
    class_count = np.bincount(train_labels)
    sample_weights = 1.0 / class_count[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader, train_full.classes, train_full.class_to_idx


def _train_one(name: str, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig) -> dict:
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), TimeElapsedColumn()) as prog:
        task_id = prog.add_task(f"Training {name}", total=None)

        def on_epoch_end(epoch: int, metrics: dict) -> None:
            prog.update(task_id, description=(
                f"{name} epoch {epoch}: train_acc={metrics['train_acc']:.3f} val_acc={metrics['val_acc']:.3f}"
            ))

        result = train(model, train_loader, val_loader, cfg, on_epoch_end=on_epoch_end)
    return {
        "history": result.history,
        "best_val_acc": result.best_val_acc,
        "best_epoch": result.best_epoch,
        "state": result.state_dict,
    }


@app.command()
def main(
    directory: Path = typer.Argument(..., exists=True),
    model: Literal["both", "scratch", "transfer"] = typer.Option("both", "--model"),
    epochs: int = typer.Option(25, "--epochs"),
    batch: int = typer.Option(32, "--batch"),
    seed: int = typer.Option(42, "--seed"),
    split: float = typer.Option(0.8, "--split"),
    out: Path = typer.Option(Path("artifacts"), "--out"),
) -> None:
    set_seed(seed)
    console.print(f"[info]Loading dataset from {directory} ...[/info]")
    train_loader, val_loader, classes, class_to_idx = _build_loaders(directory, split, batch, seed)
    n_classes = len(classes)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_val = len(val_loader.dataset)  # type: ignore[arg-type]
    console.print(f"[info]{n_train} train + {n_val} val, {n_classes} classes[/info]")
    if n_val < 100:
        die("Validation set has fewer than 100 images — PDF requires ≥ 100.")

    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict] = {}

    if model in ("both", "scratch"):
        console.print("[info]Training ScratchCNN ...[/info]")
        artifacts["scratch"] = _train_one(
            "scratch", ScratchCNN(num_classes=n_classes), train_loader, val_loader,
            TrainConfig(epochs=epochs, lr=1e-3, patience=5),
        )
        save_state(artifacts["scratch"]["state"], out / "model_scratch.pt")

    if model in ("both", "transfer"):
        console.print("[info]Training TransferModel ...[/info]")
        artifacts["transfer"] = _train_one(
            "transfer", TransferModel(num_classes=n_classes, pretrained=True), train_loader, val_loader,
            TrainConfig(epochs=epochs, lr=1e-3, patience=5, unfreeze_after=5),
        )
        save_state(artifacts["transfer"]["state"], out / "model_transfer.pt")

    # Curves + confusion matrix using the better-performing model
    best_name = max(artifacts, key=lambda k: artifacts[k]["best_val_acc"])
    learning_curves(artifacts[best_name]["history"], out / "learning_curves.png")

    # Confusion matrix on val set with best model
    best_model = ScratchCNN(num_classes=n_classes) if best_name == "scratch" else TransferModel(num_classes=n_classes, pretrained=False)
    best_model.load_state_dict(artifacts[best_name]["state"])
    best_model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            logits = best_model(x)
            y_true.extend(y.tolist())
            y_pred.extend(logits.argmax(1).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    confusion_matrix_plot(cm, classes, out / "confusion_matrix.png")
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=4)
    )

    metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "image_size": 256,
        "normalize_mean": ImageNetMean,
        "normalize_std": ImageNetStd,
        "models": {
            name: {
                "val_accuracy": round(art["best_val_acc"], 4),
                "best_epoch": art["best_epoch"],
            } for name, art in artifacts.items()
        },
        "split": {"train": split, "val": round(1 - split, 4)},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Zip artifacts
    models_zip = Path("trained_models.zip")
    with zipfile.ZipFile(models_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir():
            zf.write(p, arcname=p.name)

    aug_zip = Path("augmented_directory.zip")
    aug_dir = Path("augmented_directory")
    if aug_dir.exists():
        with zipfile.ZipFile(aug_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in aug_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(aug_dir.parent))

    sig_inputs = [models_zip] + ([aug_zip] if aug_zip.exists() else [])
    write_signature(sig_inputs, Path("signature.txt"))
    console.print("[ok]Done. trained_models.zip + signature.txt generated.[/ok]")
    for name, art in artifacts.items():
        console.print(f"  [info]{name}[/info]: best val_acc={art['best_val_acc']:.4f} @ epoch {art['best_epoch']}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: chmod + verify**

```bash
chmod +x train.py
uv run python train.py --help
```
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat(part4): train.py — two models + zip + auto signature"
```

---

### Task 19: `predict.py`

**Files:**
- Create: `predict.py`

- [ ] **Step 1: Implement**

`predict.py`:
```python
#!/usr/bin/env python3
"""Part 4 — predict disease class for a single leaf image."""
from __future__ import annotations

from pathlib import Path

import typer

from leaffliction.cli import console, die
from leaffliction.predictor import predict, render

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    image: Path = typer.Argument(..., exists=True, help="Image to classify."),
    zip_path: Path = typer.Option(Path("trained_models.zip"), "--zip", help="Trained models archive."),
    model: str = typer.Option("transfer", "--model", help="Preferred model: 'transfer' or 'scratch'."),
    save: Path | None = typer.Option(None, "--save", help="Save figure instead of showing."),
) -> None:
    if not zip_path.exists():
        die(f"Zip not found: {zip_path}. Run train.py first.")
    result = predict(image, zip_path, prefer=model)
    console.print(f"[ok]Class predicted: {result['class']} ({result['confidence']:.1%})[/ok]")
    render(result, save=save)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: chmod + verify**

```bash
chmod +x predict.py
uv run python predict.py --help
```
Expected: exit 0.

- [ ] **Step 3: Run full smoke**

```bash
uv run pytest tests/test_smoke.py -v
```
Expected: 5 PASS (all entrypoints answer --help/-h with exit 0).

- [ ] **Step 4: Commit**

```bash
git add predict.py
git commit -m "feat(part4): predict.py — load zip, render original/mask, print class"
```

---

## Phase 5 — Integration

### Task 20: Pre-commit guard against committing dataset/models

**Files:**
- Create: `scripts/check_no_dataset.sh`
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `scripts/check_no_dataset.sh`**

```bash
#!/usr/bin/env bash
# Pre-commit guard: block accidental dataset/model commits.
set -euo pipefail
forbidden_pattern='\.(zip|pt|pth)$|^images/|^augmented_directory/|^artifacts/'
violations=$(git diff --cached --name-only | grep -E "$forbidden_pattern" || true)
if [ -n "$violations" ]; then
    echo "❌ Cannot commit dataset or model files:"
    echo "$violations" | sed 's/^/   /'
    echo "💡 If unintentional, remove from index: git rm --cached <path>"
    exit 1
fi
```

- [ ] **Step 2: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: no-dataset
        name: Block dataset/model files
        entry: bash scripts/check_no_dataset.sh
        language: system
        always_run: true
        pass_filenames: false
```

- [ ] **Step 3: Install hook**

```bash
chmod +x scripts/check_no_dataset.sh
uv run pre-commit install
```
Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 4: Verify guard works**

```bash
mkdir -p augmented_directory
echo test > augmented_directory/test.JPG
git add -f augmented_directory/test.JPG
git commit -m "should fail" || echo "Guard caught it"
git reset HEAD augmented_directory/test.JPG
rm -rf augmented_directory
```
Expected: commit blocked with "Cannot commit dataset or model files".

- [ ] **Step 5: Commit hook config**

```bash
git add scripts/check_no_dataset.sh .pre-commit-config.yaml
git commit -m "chore(hooks): pre-commit guard blocking dataset/model files"
```

---

### Task 21: `scripts/verify.sh` (defense pre-flight)

**Files:**
- Create: `scripts/verify.sh`

- [ ] **Step 1: Create script**

```bash
#!/usr/bin/env bash
# Verify signature.txt against actual zip files. Used before defense.
set -euo pipefail

if [ ! -f signature.txt ]; then
    echo "❌ signature.txt not found. Run train.py first."
    exit 1
fi

cmd="shasum"
command -v "$cmd" >/dev/null || cmd="sha1sum"

failed=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    expected=$(echo "$line" | awk '{print $1}')
    name=$(echo "$line" | awk '{print $2}')
    if [ ! -f "$name" ]; then
        echo "❌ Missing file: $name"
        failed=1
        continue
    fi
    actual=$($cmd "$name" | awk '{print $1}')
    if [ "$actual" = "$expected" ]; then
        echo "✅ $name OK"
    else
        echo "❌ $name MISMATCH"
        echo "   expected: $expected"
        echo "   actual:   $actual"
        failed=1
    fi
done < signature.txt

[ "$failed" = 0 ] && echo "✅ All signatures verified." || { echo "❌ Verification failed."; exit 1; }
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x scripts/verify.sh
git add scripts/verify.sh
git commit -m "chore(scripts): verify.sh — defense-day signature check"
```

---

### Task 22: README.md (final)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# Leaffliction

> Computer vision — image classification by disease recognition on leaves (42 project).

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
uv run pre-commit install   # optional but recommended
```

## Usage

```bash
# Part 1 — dataset distribution
./Distribution.py images/

# Part 2 — single image (6 augmentations alongside) or batch balance
./Augmentation.py "images/Apple_healthy/image (1).JPG"
./Augmentation.py images/ --balance --output augmented_directory/

# Part 3 — plantCV transformations
./Transformation.py "images/Apple_healthy/image (1).JPG"
./Transformation.py -src images/Apple_healthy/ -dst out/ -mask

# Part 4 — train + predict
./train.py augmented_directory/ --epochs 25
./predict.py "images/Apple_healthy/image (1).JPG"
```

## Models

| Model | Architecture | Params | Target val_acc |
|-------|--------------|--------|----------------|
| Scratch | 4× Conv-BN-ReLU + GAP + FC | ~3.2M | 88-93% |
| Transfer | EfficientNet-B0 (ImageNet → fine-tune) | ~5.3M | 95-98% |

Both train in one `train.py` invocation; the better-performing one is used by `predict.py` by default.

## Quality

```bash
make lint        # ruff + flake8
make test        # pytest
make smoke       # entrypoint --help only
make verify      # signature.txt vs *.zip
```

## Defense Checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] `train.py` completes; both models reach val_acc ≥ 90 %
- [ ] `signature.txt`, `trained_models.zip`, `augmented_directory.zip` exist
- [ ] `make verify` confirms hashes
- [ ] `git status` shows zero `*.zip / *.pt / images/ / augmented_directory/` files

See [docs/superpowers/specs/2026-04-28-leaffliction-design.md](docs/superpowers/specs/2026-04-28-leaffliction-design.md) for full design rationale.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, usage, models, defense checklist"
```

---

### Task 23: Final integration smoke test

**Files:**
- (none — verification only)

- [ ] **Step 1: Run lint**

```bash
make lint
```
Expected: zero issues.

- [ ] **Step 2: Run tests**

```bash
make test
```
Expected: all PASS.

- [ ] **Step 3: Run a tiny end-to-end probe**

```bash
# Build a tiny 2-class dataset of 30 images each (synthetic)
python -c "
import numpy as np
from pathlib import Path
from PIL import Image
root = Path('tmp_dataset')
for cls in ('clsA', 'clsB'):
    d = root / cls; d.mkdir(parents=True, exist_ok=True)
    for i in range(30):
        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype='uint8')).save(d / f'image ({i+1}).JPG')
"

uv run python Distribution.py tmp_dataset --save tmp_dataset/dist.png
uv run python Augmentation.py "tmp_dataset/clsA/image (1).JPG"
# Skip Transformation.py probe (plantCV needs realistic leaf images)
# Skip train.py end-to-end here (long); it has its own --help test

rm -rf tmp_dataset
```
Expected: dist.png exists, six `image (1)_*.JPG` siblings exist in clsA.

- [ ] **Step 4: Final commit (clean state)**

```bash
git status
# If clean, nothing to commit. If anything changed, evaluate and commit/discard.
```

- [ ] **Step 5: Tag**

```bash
git tag -a v0.1.0 -m "Implementation complete; ready for full training run"
```

---

## Real Training Run (manual, post-implementation)

After Task 23 passes, run the actual training (this is a user task, not part of the plan because it requires the real dataset and ~30 min of GPU/MPS time):

```bash
# 1) Balance the dataset offline
./Augmentation.py images/ --balance --output augmented_directory/

# 2) Train both models
./train.py augmented_directory/ --epochs 25 --seed 42

# 3) Verify artifacts
make verify
ls -la trained_models.zip augmented_directory.zip signature.txt

# 4) Spot-check predictions
./predict.py "images/Apple_healthy/image (1).JPG"
./predict.py "images/Grape_Esca/image (1).JPG"
```

Confirm both models reach `val_acc >= 0.90` per `metadata.json`.

---

## Notes for the Executing Engineer

- **Run inside `uv` always**: `uv run python ...`. Never call system `python`.
- **plantCV install on macOS**: if `uv sync` fails on plantcv, try `uv pip install plantcv --no-build-isolation`.
- **MPS backend**: `Trainer` auto-detects. If you see "fallback to CPU", confirm `torch.backends.mps.is_available()`.
- **First training run is slow**: pretrained EfficientNet weights download once (~20 MB).
- **Don't commit `images/`, `augmented_directory/`, `*.zip`, `*.pt`**: pre-commit hook blocks; CI would too.
- **Defense day**: bring the laptop with `trained_models.zip` and `augmented_directory.zip` on disk (not in repo). Run `make verify` first thing.
