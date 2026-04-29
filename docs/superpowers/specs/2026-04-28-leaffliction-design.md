# Leaffliction — Design Specification

- **Project**: 42 Leaffliction (Computer Vision)
- **Date**: 2026-04-28
- **Status**: Approved (brainstorming phase)
- **Author**: Keonwoo Kim (keokim)

---

## 1. Goal & Constraints

### 1.1 Goal
Build an end-to-end image classification system that recognizes diseases on plant leaves, satisfying every requirement in `Leaffliction.pdf` while remaining smart, modern, and defense-safe.

### 1.2 Hard constraints (from PDF)
- Five executable programs at repository root: `Distribution.[ext]`, `Augmentation.[ext]`, `Transformation.[ext]`, `train.[ext]`, `predict.[ext]`
- Programs must not crash with segfault/bus error/double-free → otherwise grade 0
- If using Python, code must pass `flake8`
- Validation accuracy ≥ 90% on a held-out set with at least 100 images
- Defense package must include `signature.txt` (SHA1 of dataset & training zips)
- Dataset and trained models must NOT be committed to git → otherwise grade 0
- Programs must support both single-image and directory inputs where applicable
- `Transformation.[ext]` must support `-h`, `-src`, `-dst`, and individual transformation flags
- `Augmentation.[ext]` must save outputs alongside originals with the suffix convention `image (1)_Flip.JPG`, etc.

### 1.3 Quality goals
- Defense-proof: every line of code is explainable by the author
- Modern and trendy stack (PyTorch 2.x, uv, ruff, typer, Albumentations)
- Reproducible: seeded, deterministic where feasible
- Minimal artifacts in git; large files isolated to `.zip` deliverables

---

## 2. Decisions Log (with rationale)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Framework | **PyTorch 2.x vanilla + torchvision** | Industry & research standard (~90% in CV). Hand-written training loop is fully defensible. M-series Mac MPS support. No Lightning abstraction → every line is the author's. |
| 2 | Folder structure handling | **Smart parser** (handles flat, 1-level, 2-level, single-class) | Compatible with PDF examples (`./Apple`) and current flat layout (`./images`). Re-used across all five programs via `LeafDataset.discover_classes`. |
| 3 | Source layout | **Thin entrypoints + `src/leaffliction/` package** | Root has the five required scripts (PDF) but actual logic is modular and reusable. |
| 4 | Model strategy | **Two models in parallel: ScratchCNN + Transfer (EfficientNet-B0)** | Scratch demonstrates ownership; transfer secures ≥ 95% accuracy. Side-by-side comparison defeats "looks too good" suspicion. |
| 5 | Augmentation library (Part 2) | **Albumentations** | Has `Skew` and `Distortion` built in, which torchvision/PIL lack. Industry/Kaggle standard. |
| 6 | Class imbalance strategy | **Offline augmentation (PDF requirement) + WeightedRandomSampler in training** | Two-layer safety net. Offline part fulfills the PDF's "augmented_directory" requirement; online sampler stabilizes training. |
| 7 | Training-time augmentation | **torchvision.transforms.v2** (Flip / Rotation / ColorJitter / Normalize) | Light, since data is already balanced offline. |
| 8 | Package manager | **uv** | 2025-2026 standard, 10-100× faster than pip/poetry, lockfile reproducibility. Falls back to `pip install` for evaluators who don't have uv. |
| 9 | Linting | **ruff (primary) + flake8 (PDF compliance)** | Ruff for speed/dev experience; flake8 retained because the PDF explicitly mandates it. |
| 10 | CLI library | **typer** | Modern argparse replacement with auto `--help`, type-driven, rich rendering. |
| 11 | Test framework | **pytest** (lightweight, ~210 lines) | Not graded but invaluable in defense. Smoke tests for the five entrypoints + unit tests for dataset, augment, signature. |
| 12 | Reproducibility | **seed=42 across `random`, `numpy`, `torch`, `torch.mps`, `cudnn.deterministic`** | Allows re-running training to identical results. Critical for "results don't look suspicious". |

---

## 3. Architecture

### 3.1 Directory layout

```
Leaffliction/
├── Distribution.py              # Part 1 entrypoint (~30 lines, calls src/)
├── Augmentation.py              # Part 2 entrypoint
├── Transformation.py            # Part 3 entrypoint
├── train.py                     # Part 4 entrypoint
├── predict.py                   # Part 4 entrypoint
│
├── src/leaffliction/
│   ├── __init__.py
│   ├── dataset.py               # Smart folder walker (LeafDataset)
│   ├── augment.py               # Albumentations wrapper, 6 ops
│   ├── transform.py             # plantCV wrapper, 6 transformations + histogram
│   ├── viz.py                   # matplotlib/seaborn charts
│   ├── models/
│   │   ├── __init__.py
│   │   ├── scratch_cnn.py       # Hand-designed CNN (~3M params)
│   │   └── transfer.py          # EfficientNet-B0 transfer learning
│   ├── trainer.py               # Shared training loop for both models
│   ├── predictor.py             # Inference + visualization
│   ├── signature.py             # SHA1 compute + verify
│   └── cli.py                   # Shared typer helpers
│
├── tests/
│   ├── test_dataset.py
│   ├── test_augment.py
│   ├── test_signature.py
│   └── test_smoke.py
│
├── scripts/
│   ├── verify.sh                # Compare signature.txt to actual zip hashes
│   └── check_no_dataset.sh      # pre-commit guard
│
├── docs/superpowers/specs/
│   └── 2026-04-28-leaffliction-design.md
│
├── pyproject.toml               # uv + ruff + flake8 + pytest config
├── uv.lock                      # auto-generated
├── .python-version              # 3.12
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile                     # lint/format/test/train shortcuts
├── README.md
└── signature.txt                # generated by train.py, committed
```

### 3.2 Data flow

```
images/                          # raw dataset (gitignored)
   │
   ├──► Distribution.py ──► pie + bar charts (screen or PNG)
   │
   ├──► Augmentation.py (single mode)  ──► 6 variants saved alongside original
   │
   └──► Augmentation.py (batch mode)   ──► augmented_directory/  (gitignored)
                                              │
                                              ├──► Transformation.py ──► 6 plantCV
                                              │                          + color hist
                                              │
                                              └──► train.py
                                                     │
                                                     ├─► artifacts/
                                                     │     ├── model_scratch.pt
                                                     │     ├── model_transfer.pt
                                                     │     ├── metadata.json
                                                     │     ├── learning_curves.png
                                                     │     └── confusion_matrix.png
                                                     │
                                                     ├─► trained_models.zip
                                                     ├─► augmented_directory.zip
                                                     └─► signature.txt   (committed)
                                                          │
                                                          ▼
                                                    predict.py
```

### 3.3 Core abstractions

**`leaffliction.dataset.LeafDataset`** — single source of truth for data
- Constructor accepts any directory (flat, nested 1-level, nested 2-level, single class)
- Auto-discovers `class_name → list[image_path]` by finding leaf directories with images
- Implements `torch.utils.data.Dataset` so it works directly in DataLoader
- Used by all five programs for consistency

**`leaffliction.signature.compute_sha1(path) / write_signature(zips) / verify_signature(sig)`**
- `train.py` calls `write_signature` automatically at the end
- `scripts/verify.sh` calls `verify_signature` for a pre-defense sanity check

**`leaffliction.models.scratch_cnn.ScratchCNN`** — explainable, hand-designed
- 4 Conv-BN-ReLU-MaxPool blocks → GAP → Dropout → FC
- ~3M parameters
- Target: 88-93% val accuracy with augmentation and 25 epochs

**`leaffliction.models.transfer.TransferModel`** — production accuracy
- `torchvision.models.efficientnet_b0(weights=DEFAULT)`
- Replace final classifier with 8-class head
- Stage 1: freeze backbone, train head (5 epochs)
- Stage 2: unfreeze entire model, fine-tune with smaller LR (10 epochs)
- Target: 96-98% val accuracy

---

## 4. Program Specifications

### 4.1 `Distribution.py` — Part 1

**CLI**
```bash
./Distribution.py <directory>
./Distribution.py images/ --save out/distribution.png
```

**Behavior**
1. Discover classes via `LeafDataset.discover_classes(directory)`
2. Title chart based on directory basename (PDF requirement)
3. Render pie + bar charts side-by-side (matplotlib subplot, seaborn palette)
4. Display on screen by default; save PNG with `--save`
5. Friendly error if no classes found (no segfault)

### 4.2 `Augmentation.py` — Part 2

**CLI**
```bash
# Single-image mode (PDF example)
./Augmentation.py "Apple/apple_healthy/image (1).JPG"

# Batch / balance mode
./Augmentation.py images/ --balance --output augmented_directory/
```

**Single-image mode**
1. Apply six Albumentations transforms: Flip, Rotate, Skew, Shear, Crop, Distortion
2. Display 7-image grid (Original + 6 variants)
3. Save each variant alongside original with suffix: `image (1)_Flip.JPG`, etc.

**Batch mode**
1. Walk input directory, count images per class
2. Determine target count (max class size)
3. For under-represented classes, randomly sample originals and apply random transforms until target is reached
4. Copy originals + augmentations into `augmented_directory/<class>/`
5. Use `rich.progress` for visible progress bar

### 4.3 `Transformation.py` — Part 3

**CLI**
```bash
# Single image (display 6 transforms + color histogram)
./Transformation.py "Apple/apple_healthy/image (1).JPG"

# Batch mode (PDF exact form)
./Transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask

# Help
./Transformation.py -h
```

**Behavior**
1. Implement six plantCV transforms: Original, Gaussian blur, Mask, ROI objects, Analyze object, Pseudolandmarks
2. Compute color histogram across nine channels (blue, blue-yellow, green, green-magenta, hue, lightness, red, saturation, value)
3. Single-image mode: display grid + histogram
4. Batch mode: save outputs to `-dst` filtered by flag (e.g., `-mask` saves only masks)
5. Auto-detect single-image vs directory by checking the path

### 4.4 `train.py` — Part 4 (two models in one run)

**CLI**
```bash
./train.py <directory>                          # train both models in one run (default)
./train.py images/ --model scratch              # one model only
./train.py images/ --model transfer
./train.py images/ --epochs 25 --batch 32 --seed 42
```

**Behavior**
1. Load dataset → stratified 80/20 split with fixed seed
2. Apply `WeightedRandomSampler` for residual imbalance
3. Train sequentially in the same invocation (or one only when `--model` is given):
   - `ScratchCNN` first, then `TransferModel`
   - Each model gets a fresh DataLoader, same split & seed for fair comparison
4. Each epoch: log train/val loss & accuracy
5. Apply early stopping + `ReduceLROnPlateau`
6. Save artifacts:
   - `model_scratch.pt`, `model_transfer.pt` (state_dict)
   - `metadata.json`
   - `learning_curves.png`
   - `confusion_matrix.png`
   - `classification_report.txt` (per-class precision/recall/f1)
7. Zip:
   - `trained_models.zip` (contents of `artifacts/`)
   - `augmented_directory.zip` (contents of `augmented_directory/`)
8. Write `signature.txt` with both zips' SHA1
9. Print comparison table to stdout (defense talking point)

**Defense talking points generated automatically**
- Scratch CNN val accuracy vs Transfer val accuracy
- Train/val accuracy gap (small ⇒ no overfitting)
- Learning curves PNG (smooth plateau ⇒ not suspicious)

### 4.5 `predict.py` — Part 4

**CLI**
```bash
./predict.py "Apple/apple_healthy/image (1).JPG"
./predict.py path/to/image.JPG --model transfer       # default: transfer
./predict.py path/to/image.JPG --zip trained_models.zip
```

**Behavior**
1. Unzip `trained_models.zip` (or `--zip` argument) → load model + metadata
2. Preprocess image identically to training (normalize mean/std from metadata)
3. Display original + transformed image side-by-side (PDF example reproduction)
4. Print large `Class predicted : <class_name>` text on the figure
5. Append confidence score (e.g. `peach_bacterial_spot (98.3%)`)

---

## 5. Artifacts & Reproducibility

### 5.1 Generated artifacts

| File | Producer | Contents | Committed? |
|------|----------|----------|------------|
| `augmented_directory.zip` | `train.py` | All classes balanced (originals + augmentations) | ❌ |
| `trained_models.zip` | `train.py` | Two `.pt` files + `metadata.json` + curves + matrix + report | ❌ |
| `signature.txt` | `train.py` | Two SHA1 lines | ✅ |

### 5.2 `metadata.json` schema

```json
{
  "version": "1.0.0",
  "trained_at": "2026-04-28T22:35:00",
  "seed": 42,
  "classes": ["Apple_Black_rot", "Apple_healthy", "..."],
  "class_to_idx": {"Apple_Black_rot": 0},
  "image_size": 256,
  "normalize_mean": [0.485, 0.456, 0.406],
  "normalize_std": [0.229, 0.224, 0.225],
  "models": {
    "scratch":  {"val_accuracy": 0.917, "params": 3200000, "epochs": 25},
    "transfer": {"val_accuracy": 0.974, "params": 5300000, "epochs": 15}
  },
  "augmentation_balance_target": 1640,
  "split": {"train": 0.8, "val": 0.2}
}
```

### 5.3 Determinism

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 5.4 `.gitignore`

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

# OS
.DS_Store
```

### 5.5 Pre-commit guard

`.pre-commit-config.yaml` blocks any staged file matching `\.(zip|pt|pth)$`, `^images/`, or `^augmented_directory/` via `scripts/check_no_dataset.sh`.

---

## 6. Toolchain & Validation

### 6.1 `pyproject.toml` (key sections)

```toml
[project]
name = "leaffliction"
version = "1.0.0"
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

[tool.ruff]
line-length = 99
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM"]

[tool.flake8]
max-line-length = 99
extend-ignore = ["E203", "W503"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### 6.2 Test plan (~210 lines total)

| File | Tests | Approx LoC |
|------|-------|------------|
| `test_dataset.py` | discover_classes on flat / 1-level / 2-level / single-class layouts | 80 |
| `test_augment.py` | each of 6 transforms produces correct shape + dtype + saves with right filename suffix | 60 |
| `test_signature.py` | compute_sha1 matches `shasum`/`sha1sum`; verify_signature catches mismatch | 40 |
| `test_smoke.py` | each of 5 entrypoints `--help` exits 0 | 30 |

### 6.3 Makefile targets

```make
setup:    uv sync
lint:     uv run ruff check . && uv run ruff format --check . && uv run flake8 src tests *.py
format:   uv run ruff format . && uv run ruff check --fix .
test:     uv run pytest --cov=src/leaffliction
train:    uv run python train.py images/
verify:   bash scripts/verify.sh
```

### 6.4 README.md outline

- Title + summary
- Setup (`uv sync` or `pip install -r requirements.txt` fallback)
- Usage examples for all 5 programs
- Models comparison table
- Defense checklist
- Troubleshooting (M-series Mac MPS notes, plantCV install)

---

## 7. Definition of Done

- [ ] All five programs run without crash, support `-h`
- [ ] `flake8 src tests *.py` passes (PDF requirement)
- [ ] `pytest` passes
- [ ] Both models reach ≥ 90% val accuracy; transfer reaches ≥ 95%
- [ ] `train.py` automatically generates `signature.txt`, `augmented_directory.zip`, `trained_models.zip`
- [ ] `git status` shows zero dataset/model files
- [ ] `scripts/verify.sh` confirms `signature.txt` matches actual zips
- [ ] README defense checklist all checked

---

## 8. Open Questions

None at the time of approval. All five clarifying decisions captured in the Decisions Log (§2).

---

## 9. Next Step

Invoke `superpowers:writing-plans` to produce the step-by-step implementation plan.
