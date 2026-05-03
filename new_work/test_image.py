import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, ToTensord
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

# 3. Transforms (Yeh wahi hain jo training mein use hue thay, no resizing)
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")), 
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"), 
    ToTensord(keys=["image", "label"]),
])

# Splitting (Training logic)
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
print("Model loaded successfully!")

# 5. Evaluation & Visualization
dice_metric = DiceMetric(include_background=True, reduction="mean")
iou_metric = MeanIoU(include_background=True, reduction="mean")

with torch.no_grad():
    for i, batch_data in enumerate(test_loader):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        # Sliding Window Inference
        outputs = sliding_window_inference(
            inputs, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
        )
        
        outputs = torch.sigmoid(outputs) > 0.5
        dice_metric(y_pred=outputs, y=labels)
        iou_metric(y_pred=outputs, y=labels)
        
        # Save visualization of first 2 images
        if i < 2:
            mid = inputs.shape[4] // 2
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1); plt.imshow(inputs[0, 0, :, :, mid].cpu(), cmap="gray"); plt.title("Input")
            plt.subplot(1, 3, 2); plt.imshow(labels[0, 0, :, :, mid].cpu(), cmap="gray"); plt.title("Label")
            plt.subplot(1, 3, 3); plt.imshow(outputs[0, 0, :, :, mid].cpu(), cmap="gray"); plt.title("Pred")
            plt.savefig(f"debug_slice_{i}.png")
            plt.close()

print(f"Mean Dice Score: {dice_metric.aggregate().item():.4f}")
print(f"Mean IoU: {iou_metric.aggregate().item():.4f}")