"""Inference: loads a zip artifact, classifies one or many images, renders figures."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F  # noqa: N812

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


@dataclass
class LoadedArtifact:
    """Cached model + metadata so batch predict avoids reloading per image."""

    model: torch.nn.Module
    classes: list[str]
    image_size: int
    model_used: str


def load_artifact(zip_path: Path, prefer: str = "scratch") -> LoadedArtifact:
    """Unzip + load weights + classes once. Reuse across many images."""
    zip_path = Path(zip_path)
    extract_dir = zip_path.parent / f".{zip_path.stem}_unpacked"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    metadata = json.loads((extract_dir / "metadata.json").read_text())
    classes: list[str] = metadata["classes"]

    weight_file = extract_dir / f"model_{prefer}.pt"
    if not weight_file.exists():
        for alt in ("scratch", "transfer"):
            cand = extract_dir / f"model_{alt}.pt"
            if cand.exists():
                weight_file = cand
                prefer = alt
                break
    if not weight_file.exists():
        raise FileNotFoundError(f"No model_*.pt inside {zip_path}")

    model = _build_model(prefer, num_classes=len(classes))
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()

    return LoadedArtifact(
        model=model,
        classes=classes,
        image_size=int(metadata.get("image_size", 256)),
        model_used=prefer,
    )


def predict_one(artifact: LoadedArtifact, image_path: Path) -> dict:
    """Run inference on a single image using an already-loaded artifact."""
    image_path = Path(image_path)
    rgb = np.array(Image.open(image_path).convert("RGB"))
    tensor = _preprocess(rgb, size=artifact.image_size)
    with torch.no_grad():
        logits = artifact.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
    return {
        "class": artifact.classes[idx],
        "confidence": float(probs[idx]),
        "rgb": rgb,
        "transformed": mask_transform(rgb),
        "model_used": artifact.model_used,
    }


def predict(image_path: Path, zip_path: Path, prefer: str = "scratch") -> dict:
    """Single-image convenience wrapper (back-compat)."""
    artifact = load_artifact(zip_path, prefer=prefer)
    return predict_one(artifact, image_path)


def predict_many(image_paths: list[Path], zip_path: Path, prefer: str = "scratch") -> list[dict]:
    """Multi-image inference. Loads model once, iterates."""
    artifact = load_artifact(zip_path, prefer=prefer)
    return [predict_one(artifact, p) for p in image_paths]


def render(result: dict, save: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw={"wspace": 0.05})
    axes[0].imshow(result["rgb"])
    axes[0].set_title("Original", fontsize=11, pad=8)
    axes[0].axis("off")
    axes[1].imshow(result["transformed"])
    axes[1].set_title("Mask transform", fontsize=11, pad=8)
    axes[1].axis("off")

    fig.suptitle(
        f"=== DL classification ===\n"
        f"Class predicted : {result['class']} ({result['confidence']:.1%})",
        fontsize=15,
        color="#39c46a",
        y=0.985,
    )
    # rect 상단 0.86 = 상단 14%를 suptitle 전용으로 비워둠 → axes title과 겹침 방지
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    if save is not None:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
