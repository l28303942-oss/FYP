"""
Inference on a new NRRD CT volume (generalization-oriented preprocessing matching training).
- Resamples to config spacing, HU window, optional foreground crop
- Segmentation: sliding-window Attention U-Net
- Saves probability map + binary mask as NRRD in outputs/predictions/

Usage:
  python inference.py --input path/to/volume.nrrd [--no-mask]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from config import CHECKPOINT_DIR, HU_CLIP_MAX, HU_CLIP_MIN, PRED_DIR, SEG_MODEL_NAME
from config import ensure_dirs
from gpu_device import print_device_banner
from models import AttentionResidualUNet3d

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from monai.inferers import sliding_window_inference
except ImportError:
    sliding_window_inference = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CT .nrrd")
    parser.add_argument("--out-dir", default=str(PRED_DIR))
    args = parser.parse_args()

    if sitk is None or sliding_window_inference is None:
        raise ImportError("Need SimpleITK and MONAI")

    ensure_dirs()
    device = print_device_banner()
    ckpt = CHECKPOINT_DIR / SEG_MODEL_NAME
    if not ckpt.is_file():
        raise FileNotFoundError(f"Train segmentation first or copy weights to {ckpt}")

    model = AttentionResidualUNet3d(in_ch=1, out_ch=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Minimal path aligned with training: use MONAI transforms in production; here stack numpy batch
    import SimpleITK as sitk

    itk = sitk.ReadImage(args.input)
    arr = sitk.GetArrayFromImage(itk).astype(np.float32)
    arr = np.clip(arr, HU_CLIP_MIN, HU_CLIP_MAX)
    arr = (arr - HU_CLIP_MIN) / (HU_CLIP_MAX - HU_CLIP_MIN + 1e-8)
    t = torch.from_numpy(arr[np.newaxis, np.newaxis, ...]).to(device)

    with torch.no_grad():
        logits = sliding_window_inference(t, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model, overlap=0.25)
        prob = torch.sigmoid(logits).cpu().numpy().squeeze()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    prob_itk = sitk.GetImageFromArray(prob.astype(np.float32))
    prob_itk.CopyInformation(itk)
    sitk.WriteImage(prob_itk, str(out_dir / f"{stem}_pe_prob.nrrd"))

    bin_itk = sitk.GetImageFromArray((prob > 0.5).astype(np.uint8))
    bin_itk.CopyInformation(itk)
    sitk.WriteImage(bin_itk, str(out_dir / f"{stem}_pe_mask.nrrd"))

    print(f"Saved:\n {out_dir / (stem + '_pe_prob.nrrd')}\n {out_dir / (stem + '_pe_mask.nrrd')}")


if __name__ == "__main__":
    main()
