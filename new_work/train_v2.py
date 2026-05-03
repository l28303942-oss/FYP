# Hybrid PE Segmentation Training Pipeline (MONAI)

import os
import random
import numpy as np
import torch
import torch.nn as nn
from monai.transforms import LambdaD
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandShiftIntensityd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SegResNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from monai.data import decollate_batch
from monai.utils import set_determinism

# =====================================================
# FIX RANDOMNESS
# =====================================================
set_determinism(seed=42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# =====================================================
# PATHS
# =====================================================
IMG_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
MASK_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

# =====================================================
# LOAD FILES
# =====================================================
image_files = sorted([
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if f.endswith('.nrrd')
])

mask_files = sorted([
    os.path.join(MASK_DIR, f)
    for f in os.listdir(MASK_DIR)
    if f.endswith('.nrrd')
])
import SimpleITK as sitk

mask = sitk.ReadImage(mask_files[0])
mask_array = sitk.GetArrayFromImage(mask)

print(np.unique(mask_array))

data_dicts = [
    {"image": img, "label": mask}
    for img, mask in zip(image_files, mask_files)
]

random.shuffle(data_dicts)

# =====================================================
# SPLIT DATA
# =====================================================
train_size = int(0.7 * len(data_dicts))
val_size = int(0.15 * len(data_dicts))

train_files = data_dicts[:train_size]
val_files = data_dicts[train_size:train_size + val_size]
test_files = data_dicts[train_size + val_size:]

print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Test: {len(test_files)}")

# =====================================================
# TRAIN TRANSFORMS
# =====================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),

    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest")
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),

    CropForegroundd(
        keys=["image", "label"],
        source_key="image"
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=4,
        num_samples=4,
        image_key="image",
    ),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),

    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.3),
    LambdaD(keys=["label"], func=lambda x: (x > 0).astype(np.uint8)),

    EnsureTyped(keys=["image", "label"]),
])

# =====================================================
# VALIDATION TRANSFORMS
# =====================================================
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),

    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
             mode=("bilinear", "nearest")),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),

    CropForegroundd(keys=["image", "label"], source_key="image"),
    LambdaD(keys=["label"], func=lambda x: (x > 0).astype(np.uint8)),
    EnsureTyped(keys=["image", "label"]),
])

# =====================================================
# DATASETS
# =====================================================
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

# =====================================================
# 🔧 FIX 1 + 2: DATALOADER WINDOWS SAFE
# =====================================================
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,   # FIXED
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,   # FIXED
)

# =====================================================
# MODEL
# =====================================================
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    init_filters=16,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
).to(device)

# LOSS
loss_function = DiceFocalLoss(sigmoid=True, lambda_dice=0.7, lambda_focal=0.3)

# OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# SCHEDULER
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# METRICS
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(threshold=0.7)

# =====================================================
# TRAINING SETTINGS
# =====================================================
max_epochs = 100
val_interval = 2
best_metric = -1
best_metric_epoch = -1
patience = 15
patience_counter = 0

# =====================================================
# 🔧 FIX 3: MAIN GUARD + TRAIN LOOP
# =====================================================
if __name__ == "__main__":

    for epoch in range(max_epochs):

        print("-" * 50)
        print(f"Epoch {epoch + 1}/{max_epochs}")

        model.train()
        epoch_loss = 0

        for batch_data in train_loader:

            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Training Loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:

            model.eval()

            with torch.no_grad():

                for val_data in val_loader:

                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    val_outputs = sliding_window_inference(
                        val_inputs,
                        roi_size=(96, 96, 96),
                        sw_batch_size=1,
                        predictor=model,
                    )

                    val_outputs = [post_pred(post_sigmoid(i)) for i in decollate_batch(val_outputs)]
                    val_labels = decollate_batch(val_labels)

                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                print(f"Validation Dice: {metric:.4f}")

                scheduler.step(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    patience_counter = 0
                    torch.save(model.state_dict(), "best_pe_hybrid_model.pth")
                    print("Best Model Saved!")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

    print("Training Complete!")
    print(f"Best Dice: {best_metric:.4f}")
    print(f"Best Epoch: {best_metric_epoch}")