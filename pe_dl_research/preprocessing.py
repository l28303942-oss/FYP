"""
Preprocessing & splitting:
- Pair validation, corrupted removal
- Lung-region approximate bounding box (HU threshold + morphology)
- Patient-wise train/val/test split (no leakage across volumes of same patient)
- MONAI transforms with augmentation (mask-aligned)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    AUG_GAUSSIAN_STD,
    AUG_PROB_FLIP,
    AUG_PROB_INTENSITY,
    AUG_PROB_ROT90,
    HU_CLIP_MAX,
    HU_CLIP_MIN,
    IMG_DIR,
    LUNG_THRESH_HIGH,
    LUNG_THRESH_LOW,
    MASK_DIR,
    PATCH_SIZE,
    SEED,
    SPACING_MM,
    SPLIT_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
)
from config import ensure_dirs
from utils import extract_patient_id, list_nrrd_files, safe_stem, set_seed


def _try_load_pair(img_path: str, mask_path: str) -> Tuple[bool, Optional[str]]:
    import SimpleITK as sitk

    try:
        im = sitk.ReadImage(img_path)
        mk = sitk.ReadImage(mask_path)
        ia = sitk.GetArrayFromImage(im)
        ma = sitk.GetArrayFromImage(mk)
        if ia.shape != ma.shape:
            return False, "shape_mismatch"
        if not np.isfinite(ia).all():
            return False, "nan_image"
        return True, None
    except Exception as e:
        return False, str(e)


def _mask_index_ieee(masks: List[str]) -> Dict[str, str]:
    """
    Map lookup keys -> mask path for IEEE-style names: 0001RefStd.nrrd, e0032RefStd.nrrd.
    """
    index: Dict[str, str] = {}
    for m in masks:
        st = safe_stem(m)
        mo = re.match(r"^(.+?)RefStd$", st, re.I)
        if not mo:
            continue
        pref = mo.group(1)
        index[pref] = m
        if pref.isdigit():
            n = int(pref)
            for key in (pref, str(n), f"{n:04d}", f"{n:03d}"):
                index[key] = m
    return index


def _find_mask_for_image(img_stem: str, by_stem: Dict[str, str], ieee_idx: Dict[str, str]) -> Optional[str]:
    # 1) Same stem as image (demo / well-named pairs)
    if img_stem in by_stem:
        return by_stem[img_stem]
    # 2) imageStem + RefStd (e.g. e0032.nrrd -> e0032RefStd.nrrd)
    r = f"{img_stem}RefStd"
    if r in by_stem:
        return by_stem[r]
    # 3) IEEE numeric 001 <-> 0001RefStd
    if img_stem.isdigit():
        n = int(img_stem)
        for key in (img_stem, str(n), f"{n:04d}", f"{n:03d}"):
            if key in ieee_idx:
                return ieee_idx[key]
    # 4) e#### style
    if re.match(r"^e\d+$", img_stem) and img_stem in ieee_idx:
        return ieee_idx[img_stem]
    return None


def discover_pairs() -> List[Dict[str, str]]:
    imgs = list_nrrd_files(IMG_DIR)
    masks = list_nrrd_files(MASK_DIR)
    by_stem = {safe_stem(m): m for m in masks}
    ieee_idx = _mask_index_ieee(masks)
    pairs = []
    for ip in imgs:
        st = safe_stem(ip)
        mp = _find_mask_for_image(st, by_stem, ieee_idx)
        if mp is None:
            continue
        ok, err = _try_load_pair(ip, mp)
        if ok:
            pairs.append({"image": ip, "label": mp, "stem": st})
        else:
            print(f"[skip] {st}: {err}")
    return pairs


def group_pairs_by_patient(pairs: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for p in pairs:
        pid = extract_patient_id(p["stem"])
        out.setdefault(pid, []).append(p)
    return out


def split_patients_groupwise(
    grouped: Dict[str, List[Dict[str, str]]],
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Patient-wise split: all volumes of one patient stay in one fold."""
    rng = np.random.default_rng(seed)
    pids = list(grouped.keys())
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    # Integer ratios can yield 0 val/test with tiny cohorts (e.g. n=4 -> val=0).
    if n >= 3 and n_val == 0:
        if n_train > 1:
            n_train -= 1
            n_val = 1
        elif n - n_train > 1:
            n_val = 1
    train_ids = set(pids[:n_train])
    val_ids = set(pids[n_train : n_train + n_val])
    test_ids = set(pids[n_train + n_val :])

    def flat(ids):
        rows = []
        for pid in pids:
            if pid in ids:
                rows.extend(grouped[pid])
        return rows

    return flat(train_ids), flat(val_ids), flat(test_ids)


def save_split_manifest(train, val, test) -> Path:
    ensure_dirs()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / "patientwise_split.json"
    payload = {"train": train, "val": val, "test": test}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved split manifest: {path}")
    return path


def lung_bbox_ndarray(ct_hwu: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    """
    Rough lung ROI from HU range (works on loaded CT array, z,y,x).
    Returns tuple of slices or None -> caller uses full volume.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return None

    mask = (ct_hwu >= LUNG_THRESH_LOW) & (ct_hwu <= LUNG_THRESH_HIGH)
    mask = ndimage.binary_closing(mask, iterations=1)
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)
    pad = 8
    Z, Y, X = ct_hwu.shape
    zs = slice(max(0, z0 - pad), min(Z, z1 + pad + 1))
    ys = slice(max(0, y0 - pad), min(Y, y1 + pad + 1))
    xs = slice(max(0, x0 - pad), min(X, x1 + pad + 1))
    return zs, ys, xs


def build_monai_seg_transforms(
    train: bool,
    crop_foreground: bool = True,
) -> Any:
    """MONAI Compose for 3D segmentation — spacing normalize, HU window, optional lung crop via CropForeground."""
    from monai.data import ITKReader
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        Lambdad,
        LoadImaged,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        RandShiftIntensityd,
        ScaleIntensityRanged,
        Spacingd,
    )

    keys = ["image", "label"]
    itk_reader = ITKReader()

    base = [
        LoadImaged(keys=keys, reader=itk_reader),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=SPACING_MM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_CLIP_MIN,
            a_max=HU_CLIP_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if crop_foreground:
        base.append(CropForegroundd(keys=keys, source_key="image"))

    base.append(Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)))

    if train:
        aug = [
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=2,
                neg=6,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=keys, prob=AUG_PROB_FLIP, spatial_axis=0),
            RandFlipd(keys=keys, prob=AUG_PROB_FLIP, spatial_axis=1),
            RandFlipd(keys=keys, prob=AUG_PROB_FLIP, spatial_axis=2),
            RandRotate90d(keys=keys, prob=AUG_PROB_ROT90, spatial_axes=(1, 2)),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=AUG_GAUSSIAN_STD),
            RandShiftIntensityd(keys=["image"], offsets=0.12, prob=AUG_PROB_INTENSITY),
        ]
        base.extend(aug)

    base.append(EnsureTyped(keys=keys))
    return Compose(base)


def build_monai_val_seg_transforms(crop_foreground: bool = True) -> Any:
    from monai.data import ITKReader
    from monai.transforms import Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped
    from monai.transforms import Lambdad, LoadImaged, ScaleIntensityRanged, Spacingd

    keys = ["image", "label"]
    itk_reader = ITKReader()
    ts = [
        LoadImaged(keys=keys, reader=itk_reader),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=SPACING_MM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_CLIP_MIN,
            a_max=HU_CLIP_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    if crop_foreground:
        ts.append(CropForegroundd(keys=keys, source_key="image"))
    ts.append(Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)))
    ts.append(EnsureTyped(keys=keys))
    return Compose(ts)


def prepare_splits_if_needed() -> Path:
    """Discover pairs, split by patient, save manifest."""
    set_seed(SEED)
    pairs = discover_pairs()
    if not pairs:
        raise RuntimeError(
            f"No valid pairs under IMG_DIR={IMG_DIR} MASK_DIR={MASK_DIR}. Run dataset_analysis.py first."
        )
    grouped = group_pairs_by_patient(pairs)
    tr, va, te = split_patients_groupwise(grouped)
    return save_split_manifest(tr, va, te)


if __name__ == "__main__":
    prepare_splits_if_needed()
