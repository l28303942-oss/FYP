"""Classification slice dataset (shared by train + evaluation)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from config import CLS_IMG_SIZE, HU_CLIP_MAX, HU_CLIP_MIN

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class CTSlicesDataset(Dataset):
    def __init__(self, rows: List[dict], train: bool = True, seed: int = 42):
        self.rows = rows
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        if self.train:
            return max(len(self.rows) * 12, 128)
        return len(self.rows)

    def __getitem__(self, idx):
        if sitk is None:
            raise ImportError("SimpleITK required")
        row = self.rows[idx % len(self.rows)] if self.train else self.rows[idx]
        ia = sitk.GetArrayFromImage(sitk.ReadImage(row["image"])).astype(np.float32)
        ma = sitk.GetArrayFromImage(sitk.ReadImage(row["label"])).astype(np.float32)
        Z = ia.shape[0]
        z = int(self.rng.integers(0, Z)) if self.train else Z // 2
        sl_img = ia[z]
        sl_mk = ma[z]
        sl_img = np.clip(sl_img, HU_CLIP_MIN, HU_CLIP_MAX)
        sl_img = (sl_img - HU_CLIP_MIN) / (HU_CLIP_MAX - HU_CLIP_MIN + 1e-8)
        y = 1 if np.any(sl_mk > 0) else 0
        if self.train:
            jit = self.rng.uniform(-0.06, 0.06)
            sl_img = np.clip(sl_img + jit, 0, 1)
            if self.rng.random() < 0.15:
                sl_img = np.clip(sl_img + self.rng.normal(0, 0.02, sl_img.shape).astype(np.float32), 0, 1)
        t = torch.from_numpy(sl_img).unsqueeze(0)
        t = TF.resize(t, [CLS_IMG_SIZE, CLS_IMG_SIZE], antialias=True).repeat(3, 1, 1)
        return t.float(), torch.tensor(y, dtype=torch.long)

