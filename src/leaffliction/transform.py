"""plantCV transformations + 9-channel color histogram (PDF Part 3)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plantcv import plantcv as pcv

TRANSFORM_NAMES = (
    "Original",
    "GaussianBlur",
    "Mask",
    "RoiObjects",
    "AnalyzeObject",
    "Pseudolandmarks",
)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def _binary_mask(rgb: np.ndarray) -> np.ndarray:
    """Leaf segmentation tuned for PlantVillage-style controlled images.

    Pipeline:
    1. LAB chroma magnitude — distance from neutral grey in the a*/b*
       plane: sqrt((a-128)² + (b-128)²). Grey background ≈ 0; green
       leaves and brown lesions both have positive chroma.
    2. Otsu auto-threshold on the chroma image.
    3. Morphological opening removes isolated background speckle.
    4. Keep only the largest connected component — the leaf — which
       drops any residual scattered blobs in the background.
    5. Fill internal holes (textured surface, light specular spots,
       and incidentally the natural holes in heavily diseased leaves);
       we accept that trade-off because the "true" leaf footprint is
       its outer boundary, not the pinholes inside it.
    """
    from scipy.ndimage import binary_fill_holes

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128
    chroma = np.sqrt(a * a + b * b).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    filled = binary_fill_holes(binary > 0).astype(np.uint8) * 255
    return pcv.fill(bin_img=filled, size=200)


def gaussian_blur(rgb: np.ndarray) -> np.ndarray:
    """Gaussian-blurred binary leaf mask (uses the same segmentation).

    Uses cv2.GaussianBlur directly rather than pcv.gaussian_blur, which
    in plantCV 4.x can interfere with matplotlib's current figure state
    (it's designed for plantCV's debug-plot workflow). cv2 keeps the
    output as a clean 0–255 uint8 array suitable for plt.imshow.
    """
    return cv2.GaussianBlur(_binary_mask(rgb), (5, 5), 0)


def mask(rgb: np.ndarray) -> np.ndarray:
    """Apply binary mask to original; background white."""
    cleaned = _binary_mask(rgb)
    return pcv.apply_mask(img=rgb, mask=cleaned, mask_color="white")


def roi_objects(rgb: np.ndarray) -> np.ndarray:
    """Return RGB with leaf contour outlined in green + ROI box in blue."""
    cleaned = _binary_mask(rgb)
    out = rgb.copy()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    h, w = rgb.shape[:2]
    cv2.rectangle(out, (5, 5), (w - 5, h - 5), (0, 0, 255), 2)
    return out


def analyze_object(rgb: np.ndarray) -> np.ndarray:
    """Run shape analysis and return its visualization."""
    cleaned = _binary_mask(rgb)
    # plantCV 4.x API may have changed. Try analyze.size first, fall back gracefully.
    try:
        labeled, _ = pcv.create_labels(mask=cleaned, rois=None, roi_type="partial")
        pcv.analyze.size(img=rgb, labeled_mask=labeled, n_labels=1)
        if pcv.outputs.images:
            return np.array(pcv.outputs.images[-1])
    except Exception:  # noqa: BLE001
        pass
    # Fallback: draw centroid + bounding box manually using cv2
    out = rgb.copy()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 2)
        moments = cv2.moments(c)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(out, (cx, cy), 4, (255, 0, 255), -1)
    return out


def pseudolandmarks(rgb: np.ndarray) -> np.ndarray:
    """Place pseudolandmark points on leaf perimeter."""
    cleaned = _binary_mask(rgb)
    out = rgb.copy()
    # plantCV 4.x: try x_axis_pseudolandmarks; fall back to evenly-spaced contour samples
    try:
        top, bottom, center = pcv.homology.x_axis_pseudolandmarks(
            img=rgb, mask=cleaned, label="leaf"
        )
        for color, points in (
            ((0, 0, 255), top),
            ((255, 165, 0), center),
            ((255, 0, 255), bottom),
        ):
            arr = np.atleast_2d(np.asarray(points)).reshape(-1, 2).astype(int)
            for pt in arr:
                cv2.circle(out, tuple(pt), 3, color, -1)
        return out
    except Exception:  # noqa: BLE001
        pass
    # Fallback: sample contour evenly and color top/middle/bottom rings differently
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    n = len(contour)
    if n < 30:
        return out
    samples = contour[:: max(1, n // 30)]
    h_img = rgb.shape[0]
    for x, y in samples:
        if y < h_img / 3:
            color = (0, 0, 255)
        elif y < 2 * h_img / 3:
            color = (255, 165, 0)
        else:
            color = (255, 0, 255)
        cv2.circle(out, (int(x), int(y)), 3, color, -1)
    return out


def color_histogram(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return {channel_name: 256-bin histogram (% of pixels)} for 9 channels."""
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
    """Run all 6 PDF-required transforms, returning a dict keyed by name."""
    return {
        "Original": rgb,
        "GaussianBlur": gaussian_blur(rgb),
        "Mask": mask(rgb),
        "RoiObjects": roi_objects(rgb),
        "AnalyzeObject": analyze_object(rgb),
        "Pseudolandmarks": pseudolandmarks(rgb),
    }
