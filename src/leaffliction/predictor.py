"""Inference: loads a zip artifact, classifies one image, renders a figure."""
from __future__ import annotations

import json
import zipfile
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


def predict(image_path: Path, zip_path: Path, prefer: str = "transfer") -> dict:
    """Return {'class', 'confidence', 'rgb', 'transformed', 'model_used'}."""
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
        for alt in ("transfer", "scratch"):
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
        "model_used": prefer,
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
        f"=== DL classification ===\n"
        f"Class predicted : {result['class']} ({result['confidence']:.1%})",
        fontsize=14,
        color="#39c46a",
    )
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
