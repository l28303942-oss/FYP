"""
Global configuration for PE deep learning research pipeline.
Edit paths to match your machine. Default: IEEE NRRD dataset layout.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Dataset (user-provided) ---
BASE = Path(r"D:\my_project\Dataset And Files\IEEE dataset")
IMG_DIR = os.environ.get("PE_IMG_DIR", str(BASE / "images" / "images"))
MASK_DIR = os.environ.get("PE_MASK_DIR", str(BASE / "rs" / "rs"))

# --- Outputs (default on D: so large checkpoints/reports don't fill C:) ---
_DEFAULT_OUTPUT_ROOT = Path(r"D:\my_project\pe_dl_outputs")
OUT_DIR = Path(os.environ.get("PE_OUT_DIR", str(_DEFAULT_OUTPUT_ROOT)))
SPLIT_DIR = OUT_DIR / "splits"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
PRED_DIR = OUT_DIR / "predictions"

# --- Reproducibility ---
SEED = 42

# --- CT / lung window (Hounsfield) ---
HU_CLIP_MIN = -1000
HU_CLIP_MAX = 400
LUNG_THRESH_LOW = -900
LUNG_THRESH_HIGH = -200

# --- 3D patch / spacing ---
SPACING_MM = (1.0, 1.0, 1.0)  # resample to isotropic 1mm (adjust if anisotropic preferred)
PATCH_SIZE = tuple(int(x) for x in os.environ.get("PE_PATCH", "96,96,96").split(","))
SW_BATCH_SIZE = int(os.environ.get("PE_SW_BATCH", "2"))  # sliding-window batch (lower if OOM on 8GB)

# --- Training segmentation ---
# ~8GB VRAM: keep SEG_BATCH_SIZE at 1–2 for 96³ patches; use AMP (default on GPU)
SEG_BATCH_SIZE = int(os.environ.get("PE_SEG_BATCH", "2"))
SEG_MAX_EPOCHS = int(os.environ.get("PE_SEG_EPOCHS", "150"))
SEG_LR = 1e-4
SEG_WEIGHT_DECAY = 1e-5
SEG_VAL_INTERVAL = 1
# PE_SEG_EARLY_STOP=0 / false / off -> run all PE_SEG_EPOCHS (no early exit)
_es = os.environ.get("PE_SEG_EARLY_STOP", "1").strip().lower()
if _es in ("0", "false", "no", "off"):
    SEG_EARLY_STOP_PATIENCE = SEG_MAX_EPOCHS + 10_000
else:
    SEG_EARLY_STOP_PATIENCE = int(os.environ.get("PE_SEG_PATIENCE", "20"))
# Mixed precision on GPU (saves VRAM). PE_AMP=0 to disable.
SEG_USE_AMP = os.environ.get("PE_AMP", "1") not in ("0", "false", "False")

# --- Training classification ---
CLS_BATCH_SIZE = int(os.environ.get("PE_CLS_BATCH", "8"))
CLS_MAX_EPOCHS = int(os.environ.get("PE_CLS_EPOCHS", "80"))
CLS_LR = 3e-4
CLS_WEIGHT_DECAY = 1e-4
CLS_IMG_SIZE = 224  # 2D EfficientNet
CLS_SLICES_PER_VOL = 24  # sample slices per epoch per volume (balanced)

# --- Split ratios (patient-wise) ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Augmentation ---
AUG_PROB_FLIP = 0.5
AUG_PROB_ROT90 = 0.3
AUG_PROB_INTENSITY = 0.3
AUG_GAUSSIAN_STD = 0.02

# --- Loss tuning (reduce false positives on segmentation) ---
DICE_FOCAL_LAMBDA_DICE = 0.65
DICE_FOCAL_LAMBDA_FOCAL = 0.35
SEG_PRED_THRESHOLD = 0.5

# --- Model filenames ---
SEG_MODEL_NAME = "best_seg_attention_unet3d.pth"
CLS_MODEL_NAME = "best_cls_efficientnet.pth"


def ensure_dirs() -> None:
    for d in (OUT_DIR, SPLIT_DIR, CHECKPOINT_DIR, FIG_DIR, REPORT_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)
