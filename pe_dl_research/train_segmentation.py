"""
Train Attention Residual U-Net for PE segmentation.
- Patient-wise splits from preprocessing.prepare_splits_if_needed()
- Dice + Focal loss, AdamW, cosine/plateau scheduler, AMP, early stopping
- Saves best weights + loss/dice curves
Run:  python train_segmentation.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    CHECKPOINT_DIR,
    DICE_FOCAL_LAMBDA_DICE,
    DICE_FOCAL_LAMBDA_FOCAL,
    FIG_DIR,
    PATCH_SIZE,
    SEG_BATCH_SIZE,
    SEG_EARLY_STOP_PATIENCE,
    SEG_LR,
    SEG_MAX_EPOCHS,
    SEG_MODEL_NAME,
    SEG_PRED_THRESHOLD,
    SEG_USE_AMP,
    SEG_VAL_INTERVAL,
    SEG_WEIGHT_DECAY,
    SPLIT_DIR,
    SW_BATCH_SIZE,
)
from config import ensure_dirs
from gpu_device import autocast_if_cuda, make_grad_scaler, print_device_banner
from models import AttentionResidualUNet3d
from preprocessing import build_monai_seg_transforms, build_monai_val_seg_transforms, prepare_splits_if_needed
from utils import set_seed

try:
    from monai.data import DataLoader, Dataset
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceFocalLoss
    from monai.metrics import DiceMetric
    from monai.transforms import Activations, AsDiscrete
    from monai.data import decollate_batch
except ImportError as e:
    raise ImportError("Install MONAI: pip install monai") from e


def load_split():
    split_path = SPLIT_DIR / "patientwise_split.json"
    if not split_path.is_file():
        prepare_splits_if_needed()
    with open(split_path, encoding="utf-8") as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]


def train_one_epoch(model, loader, loss_fn, opt, scaler, device, amp):
    model.train()
    losses = []
    use_amp = amp and device.type == "cuda"
    cm = autocast_if_cuda(device)
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with cm:
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate_volume(model, val_files, val_transform, device, dice_metric, post_sigmoid, post_pred):
    model.eval()
    if not val_files:
        return 0.0
    ds = Dataset(data=val_files, transform=val_transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    dice_metric.reset()
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = sliding_window_inference(
            x,
            roi_size=PATCH_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=0.25,
        )
        preds = [post_pred(post_sigmoid(i)) for i in decollate_batch(logits)]
        ys = decollate_batch(y)
        dice_metric(y_pred=preds, y=ys)
    agg = dice_metric.aggregate()
    if agg is None:
        return 0.0
    return agg.item()


def main():
    set_seed(42)
    ensure_dirs()
    device = print_device_banner()
    train_files, val_files, _ = load_split()
    if not train_files:
        raise RuntimeError("Train split empty — check dataset paths.")

    train_tf = build_monai_seg_transforms(train=True)
    val_tf = build_monai_val_seg_transforms()

    train_ds = Dataset(data=train_files, transform=train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=SEG_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = AttentionResidualUNet3d(in_ch=1, out_ch=1, base=16, dropout_p=0.15).to(device)

    loss_fn = DiceFocalLoss(
        sigmoid=True,
        lambda_dice=DICE_FOCAL_LAMBDA_DICE,
        lambda_focal=DICE_FOCAL_LAMBDA_FOCAL,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=SEG_LR, weight_decay=SEG_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(SEG_MAX_EPOCHS, 10))
    scaler = make_grad_scaler(device, SEG_USE_AMP)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=SEG_PRED_THRESHOLD)

    best_dice = -1.0
    patience_cnt = 0
    hist_train_loss = []
    hist_val_dice = []

    for epoch in range(1, SEG_MAX_EPOCHS + 1):
        tl = train_one_epoch(model, train_loader, loss_fn, opt, scaler, device, SEG_USE_AMP)
        hist_train_loss.append(tl)
        scheduler.step()

        if epoch % SEG_VAL_INTERVAL == 0:
            vd = validate_volume(model, val_files, val_tf, device, dice_metric, post_sigmoid, post_pred)
            hist_val_dice.append(vd)
            print(f"Epoch {epoch:03d}  train_loss={tl:.4f}  val_dice={vd:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
            if vd > best_dice:
                best_dice = vd
                patience_cnt = 0
                ckpt = CHECKPOINT_DIR / SEG_MODEL_NAME
                torch.save(model.state_dict(), ckpt)
                print(f"  -> saved best to {ckpt}")
            else:
                patience_cnt += 1
                if patience_cnt >= SEG_EARLY_STOP_PATIENCE:
                    print("Early stopping.")
                    break
        else:
            print(f"Epoch {epoch:03d}  train_loss={tl:.4f}")

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist_train_loss)
    ax[0].set_title("Train loss")
    ax[1].plot(hist_val_dice)
    ax[1].set_title("Val Dice")
    fig.tight_layout()
    fig_path = FIG_DIR / "seg_training_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure: {fig_path}")
    print(f"Best val Dice (approx): {best_dice:.4f}")


if __name__ == "__main__":
    main()
