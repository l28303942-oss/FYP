"""Create tiny synthetic CT+mask NRRD pairs for smoke-testing the pipeline when IEEE folder is empty."""
from __future__ import annotations

import os

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    raise SystemExit("pip install SimpleITK")

OUT = os.path.join(os.path.dirname(__file__), "demo_nrrd")
IMG = os.path.join(OUT, "images")
MSK = os.path.join(OUT, "masks")
os.makedirs(IMG, exist_ok=True)
os.makedirs(MSK, exist_ok=True)

rng = np.random.default_rng(42)

for i in range(4):
    z, y, x = 64, 64, 64
    ct = rng.normal(-600, 150, (z, y, x)).astype(np.float32)
    # fake PE blob
    zz, yy, xx = np.ogrid[:z, :y, :x]
    cx, cy, cz = z // 2 + i * 2, y // 2, x // 2
    dist = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    mask = (dist < (8 + i) ** 2).astype(np.uint8)
    ct[mask > 0] += 400  # brighter "clot"
    # stem prefix = patient id for group splits (pat00, pat01, ...)
    name = f"pat{i:02d}_vol.nrrd"
    itk_i = sitk.GetImageFromArray(ct)
    itk_m = sitk.GetImageFromArray(mask)
    itk_i.SetSpacing((1.0, 1.0, 1.0))
    itk_m.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(itk_i, os.path.join(IMG, name))
    sitk.WriteImage(itk_m, os.path.join(MSK, name))

print("Wrote demo volumes to:", OUT)
print("Set environment before training:")
print(f'  set PE_IMG_DIR={IMG}')
print(f'  set PE_MASK_DIR={MSK}')
