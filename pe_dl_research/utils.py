"""Shared utilities: seeding, patient IDs, logging, I/O helpers."""
from __future__ import annotations

import os
import random
import re
from typing import List

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def extract_patient_id(stem: str) -> str:
    """
    Map filename stem to a patient / study key for group-wise splits.
    Heuristics: take leading token before first double-underscore, or
    first 3 components of DICOM-style names, else full stem.
    """
    s = stem.strip()
    if "__" in s:
        return s.split("__", 1)[0]
    # e.g. 1.2.840.113... long UID -> use first 3 dot parts
    if s.replace(".", "").isdigit() or s.count(".") > 2:
        parts = s.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
    # default: first underscore segment (patient12_slice1 -> patient12)
    if "_" in s:
        return s.split("_")[0]
    return s


def list_nrrd_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".nrrd", ".nii", ".nii.gz"))
    )


def safe_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace(".nii", "")
