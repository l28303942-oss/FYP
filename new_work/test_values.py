import os
import torch
import numpy as np
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, ToTensord, KeepLargestConnectedComponent
)
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference

# 1. Device aur Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
MASK_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

# 2. Data Preparation
image_files = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.nrrd')])
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nrrd')])
data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

# 3. Transforms
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")), 
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"), 
    ToTensord(keys=["image", "label"]),
])

# Splitting
train_size = int(0.7 * len(data_dicts))
val_size = int(0.15 * len(data_dicts))
test_files = data_dicts[train_size + val_size :]

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

# 4. Model Load
model = AttentionUnet(
    spatial_dims=3, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
).to(device)

model.load_state_dict(torch.load("best_simple_model.pth", map_location=device))
model.eval()

# 5. Metrics & Post-processing Setup
dice_metric = DiceMetric(include_background=True, reduction="mean")
iou_metric = MeanIoU(include_background=True, reduction="mean")
# KeepLargestConnectedComponent: Ye noise aur chhote fuzool spots ko hata deta hai
keep_largest = KeepLargestConnectedComponent(applied_labels=[1], is_onehot=False)

print("Running testing with noise removal...")

with torch.no_grad():
    for batch_data in test_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        # Sliding Window Inference
        outputs = sliding_window_inference(
            inputs, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
        )
        
        # Step 1: Thresholding (Confidence barhai, 0.85 tak)
        outputs = (torch.sigmoid(outputs) > 0.85).float()
        
        # Step 2: Noise Removal (KeepLargestConnectedComponent)
        outputs = keep_largest(outputs)
        
        dice_metric(y_pred=outputs, y=labels)
        iou_metric(y_pred=outputs, y=labels)

print(f"Mean Dice Score: {dice_metric.aggregate().item():.4f}")
print(f"Mean IoU: {iou_metric.aggregate().item():.4f}")