"""
Automatic dataset structure analysis: pair image/mask NRRD volumes,
flag corrupted or mismatched files, class balance, shape stats.
Run:  python -m pe_dl_research.dataset_analysis
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import IMG_DIR, MASK_DIR, REPORT_DIR, ensure_dirs
from utils import extract_patient_id, list_nrrd_files, safe_stem, set_seed

set_seed(42)


def try_read_sitk(path: str):
    import SimpleITK as sitk

    try:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        return True, arr.shape, float(np.nanmin(arr)), float(np.nanmax(arr)), None
    except Exception as e:
        return False, None, None, None, str(e)


def analyze() -> dict:
    ensure_dirs()
    imgs = list_nrrd_files(IMG_DIR)
    masks = list_nrrd_files(MASK_DIR)

    mask_by_stem = {safe_stem(m): m for m in masks}
    pairs = []
    orphans_img = []
    for ip in imgs:
        st = safe_stem(ip)
        if st in mask_by_stem:
            pairs.append({"image": ip, "mask": mask_by_stem[st], "stem": st})
        else:
            orphans_img.append(ip)

    orphans_mask = [m for m in masks if safe_stem(m) not in {safe_stem(i) for i in imgs}]

    corrupted = []
    shapes_img = []
    shapes_mask = []
    label_stats = []
    patients = defaultdict(int)

    for p in pairs:
        ok_i, sh_i, mn_i, mx_i, er_i = try_read_sitk(p["image"])
        ok_m, sh_m, mn_m, mx_m, er_m = try_read_sitk(p["mask"])
        if not ok_i or not ok_m:
            corrupted.append(
                {
                    "stem": p["stem"],
                    "image_err": er_i,
                    "mask_err": er_m,
                }
            )
            continue
        if sh_i != sh_m:
            corrupted.append(
                {
                    "stem": p["stem"],
                    "shape_mismatch": {"image": sh_i, "mask": sh_m},
                }
            )
            continue

        shapes_img.append(sh_i)
        shapes_mask.append(sh_m)
        # reload mask for stats
        import SimpleITK as sitk

        m_arr = sitk.GetArrayFromImage(sitk.ReadImage(p["mask"]))
        pos = float(np.mean(m_arr > 0))
        label_stats.append(pos)
        pid = extract_patient_id(p["stem"])
        patients[pid] += 1

    report = {
        "image_dir": IMG_DIR,
        "mask_dir": MASK_DIR,
        "num_images_found": len(imgs),
        "num_masks_found": len(masks),
        "num_valid_pairs": len(pairs) - len(corrupted),
        "num_pairs_listed": len(pairs),
        "orphan_images": len(orphans_img),
        "orphan_masks": len(orphans_mask),
        "corrupted_or_mismatch": corrupted,
        "unique_patients_estimated": len(patients),
        "patient_volume_counts": dict(patients),
        "foreground_fraction_mean": float(np.mean(label_stats)) if label_stats else None,
        "foreground_fraction_std": float(np.std(label_stats)) if label_stats else None,
    }

    if shapes_img:
        report["typical_image_shape"] = shapes_img[0]
        report["num_shapes_unique"] = len(set(shapes_img))

    OUT_JSON = REPORT_DIR / "dataset_analysis.json"
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT_JSON}")

    if corrupted:
        print("\n[WARNING] Fix corrupted/mismatched pairs before training:")
        for c in corrupted[:10]:
            print(" ", c)
    if not pairs:
        print(
            "\n[NOTE] No pairs found. Check IMG_DIR / MASK_DIR paths and that filenames match (same stem)."
        )
    return report


if __name__ == "__main__":
    analyze()
