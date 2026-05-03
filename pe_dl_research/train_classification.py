"""
Slice-level PE classification (EfficientNet-B0).
Run: python train_classification.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from cls_data import CTSlicesDataset
from config import (
    CHECKPOINT_DIR,
    CLS_BATCH_SIZE,
    CLS_LR,
    CLS_MAX_EPOCHS,
    CLS_MODEL_NAME,
    CLS_WEIGHT_DECAY,
    SEG_USE_AMP,
    SPLIT_DIR,
    ensure_dirs,
)
from gpu_device import autocast_if_cuda, make_grad_scaler, print_device_banner
from models import EfficientNetPEClassifier
from preprocessing import prepare_splits_if_needed
from utils import set_seed


def main():
    set_seed(42)
    ensure_dirs()
    device = print_device_banner()
    split_path = SPLIT_DIR / "patientwise_split.json"
    if not split_path.is_file():
        prepare_splits_if_needed()
    with open(split_path, encoding="utf-8") as f:
        sp = json.load(f)
    train_rows, val_rows, _ = sp["train"], sp["val"], sp["test"]

    train_ds = CTSlicesDataset(train_rows, train=True)
    val_ds = CTSlicesDataset(val_rows, train=False)

    w = torch.ones(len(train_rows), dtype=torch.double)
    sampler = WeightedRandomSampler(w, num_samples=min(len(train_rows) * 24, 8192), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CLS_BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=CLS_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = EfficientNetPEClassifier(num_classes=2, dropout=0.35).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CLS_LR, weight_decay=CLS_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    scaler = make_grad_scaler(device, SEG_USE_AMP)
    cm = autocast_if_cuda(device)
    use_amp = SEG_USE_AMP and device.type == "cuda"

    best_acc = 0.0
    for epoch in range(1, CLS_MAX_EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with cm:
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
            losses.append(loss.item())

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(1, total)
        sched.step(acc)
        print(f"Epoch {epoch:03d}  loss={np.mean(losses):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / CLS_MODEL_NAME)
            print(f"  saved {CLS_MODEL_NAME}")

    print(f"Best val acc ~ {best_acc:.4f}")


if __name__ == "__main__":
    main()
