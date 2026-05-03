"""
Inference pipeline: preprocess CT slice → segmentation mask + overlays + confidence.
Falls back to deterministic demo segmentation if PyTorch model cannot load.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


@dataclass
class InferenceResult:
    mask_array: np.ndarray  # H,W float 0-1
    confidence: float
    processing_time_sec: float
    used_real_model: bool
    message: str


def _load_image_gray(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    arr = None

    if ext in (".dcm", ".dicom"):
        try:
            import pydicom

            ds = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
        except Exception:
            arr = None

    if arr is None:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr


def _demo_segmentation(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Anatomically meaningless demo mask for UI when no weights load."""
    h, w = gray.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = np.clip(1.0 - np.abs(dist - min(h, w) * 0.22) / (min(h, w) * 0.08), 0, 1)
    edges = np.abs(np.gradient(gray.astype(np.float64))[0]) + np.abs(np.gradient(gray.astype(np.float64))[1])
    edges = edges / (edges.max() + 1e-6)
    mask = np.clip(ring * 0.6 + edges * 0.5 + gray * 0.15, 0, 1)
    mask = (mask > 0.45).astype(np.float32)
    conf = float(0.72 + 0.08 * np.sin(h * 0.01) + 0.05 * np.mean(gray))
    conf = float(np.clip(conf, 0.55, 0.93))
    return mask, conf


def _torch_predict(
    gray: np.ndarray,
    weights_path: str | None,
    device: str,
) -> Tuple[np.ndarray, float, bool, str]:
    if not _TORCH:
        m, c = _demo_segmentation(gray)
        return m, c, False, "PyTorch not installed; demo mask used."

    from ml.attention_unet import build_model

    model, loaded = build_model(in_channels=1, weights_path=weights_path, device=device)
    if not loaded:
        m, c = _demo_segmentation(gray)
        return m, c, False, "Model weights missing or incompatible; demo segmentation applied."

    # Resize to multiple of 16
    h, w = gray.shape[:2]
    th, tw = max(256, h - h % 16), max(256, w - w % 16)
    img = Image.fromarray((gray * 255).astype(np.uint8))
    img = img.resize((tw, th), Image.BILINEAR)
    t = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(t).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (prob > 0.5).astype(np.float32)
    conf = float(np.clip(prob.max(), 0.1, 0.99))
    # Resize mask back
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((w, h), Image.NEAREST)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return mask, conf, True, "Attention U-Net inference complete."


def run_inference(
    image_path: str,
    weights_path: str | None,
    device: str | None = None,
) -> InferenceResult:
    t0 = time.perf_counter()
    gray = _load_image_gray(image_path)
    if gray.ndim != 2:
        gray = np.mean(gray, axis=-1)

    dev = device or ("cuda" if _TORCH and __import__("torch").cuda.is_available() else "cpu")

    mask, conf, real, msg = _torch_predict(gray, weights_path, dev)
    elapsed = time.perf_counter() - t0

    return InferenceResult(
        mask_array=mask,
        confidence=conf,
        processing_time_sec=elapsed,
        used_real_model=real,
        message=msg,
    )


def save_visualizations(
    original_path: str,
    mask: np.ndarray,
    out_dir: str,
    base_name: str,
) -> Tuple[str, str, str]:
    """Save mask PNG, heatmap, overlay; return paths."""
    os.makedirs(out_dir, exist_ok=True)
    gray = _load_image_gray(original_path)
    if gray.ndim != 2:
        gray = np.mean(gray, axis=-1)
    h, w = gray.shape[:2]
    if mask.shape != (h, w):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((w, h), Image.NEAREST)
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0

    mask_path = os.path.join(out_dir, f"{base_name}_mask.png")
    heat_path = os.path.join(out_dir, f"{base_name}_heatmap.png")
    overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")

    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

    heat = (mask * np.linspace(0.2, 1.0, w)).astype(np.float32)
    heat_rgb = np.stack([heat, mask * 0.3, 1.0 - mask], axis=-1)
    heat_rgb = (heat_rgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(heat_rgb).save(heat_path)

    base_rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = base_rgb.copy()
    red = np.array([1.0, 0.15, 0.15])
    for c in range(3):
        overlay[..., c] = np.clip(overlay[..., c] * (1 - mask * 0.65) + mask * red[c] * 0.85, 0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)

    return mask_path, heat_path, overlay_path
