"""
Evaluate segmentation + classification on test split.
Writes metrics JSON + confusion matrix PNG + optional ROC data.
Run: python evaluation.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve

from config import CHECKPOINT_DIR, CLS_MODEL_NAME, REPORT_DIR, SEG_MODEL_NAME, SPLIT_DIR, SEG_PRED_THRESHOLD
from config import ensure_dirs
from gpu_device import print_device_banner
from models import AttentionResidualUNet3d, EfficientNetPEClassifier
from preprocessing import build_monai_val_seg_transforms
from utils import set_seed

try:
    from monai.data import DataLoader, Dataset
    from monai.inferers import sliding_window_inference
except ImportError:
    raise ImportError("pip install monai")

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def seg_metrics_batch(pred_bin: np.ndarray, gt_bin: np.ndarray):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    iou = inter / (union + 1e-8)
    dice = (2 * inter) / (pred_sum + gt_sum + 1e-8)
    prec = inter / (pred_sum + 1e-8)
    rec = inter / (gt_sum + 1e-8)
    return dict(dice=float(dice), iou=float(iou), precision=float(prec), recall=float(rec))


def evaluate_segmentation(device: torch.device, test_files: list):
    if not test_files:
        return {}
    val_tf = build_monai_val_seg_transforms()
    ds = Dataset(data=test_files, transform=val_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ckpt = CHECKPOINT_DIR / SEG_MODEL_NAME
    if not ckpt.is_file():
        print(f"Missing checkpoint {ckpt}; train segmentation first.")
        return {}

    model = AttentionResidualUNet3d(in_ch=1, out_ch=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=SEG_PRED_THRESHOLD)

    agg = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].cpu().numpy()
            logits = sliding_window_inference(
                x,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > SEG_PRED_THRESHOLD).astype(np.uint8)
            for i in range(pred.shape[0]):
                m = seg_metrics_batch(pred[i].squeeze() > 0, y[i].squeeze() > 0)
                agg.append(m)

    mean = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
    return {"per_volume": agg, "mean": mean}


def evaluate_classification(device: torch.device, test_rows: list):
    if sitk is None or not test_rows:
        return {}
    ckpt = CHECKPOINT_DIR / CLS_MODEL_NAME
    if not ckpt.is_file():
        print("Missing classification checkpoint.")
        return {}

    model = EfficientNetPEClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ys, ps = [], []
    from cls_data import CTSlicesDataset

    tmp = CTSlicesDataset(test_rows, train=False)
    loader = torch.utils.data.DataLoader(tmp, batch_size=8, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())
            probs_list.extend(prob.tolist())

    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="binary", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=[0, 1])
    out = {"accuracy": float(acc), "f1_binary": float(f1), "confusion_matrix": cm.tolist()}
    try:
        fpr, tpr, _ = roc_curve(ys, probs_list)
        out["auc"] = float(auc(fpr, tpr)) if len(np.unique(ys)) > 1 else None
    except Exception:
        out["auc"] = None

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Classification confusion matrix")
    fig.savefig(REPORT_DIR / "cls_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    set_seed(42)
    ensure_dirs()
    sp_path = SPLIT_DIR / "patientwise_split.json"
    if not sp_path.is_file():
        print("Run preprocessing.prepare_splits_if_needed first.")
        return
    with open(sp_path, encoding="utf-8") as f:
        sp = json.load(f)
    test_files = sp["test"]
    device = print_device_banner()

    seg_res = evaluate_segmentation(device, test_files)
    cls_res = evaluate_classification(device, test_files)

    report = {"segmentation_test": seg_res, "classification_test": cls_res}
    out_p = REPORT_DIR / "evaluation_report.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
