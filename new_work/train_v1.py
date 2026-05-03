import os
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, CropForegroundd, 
    RandCropByPosNegLabeld, ToTensord
)
from monai.data import DataLoader, Dataset

# 1. Paths set karein
IMG_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
MASK_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

# 2. Files ki list aur Mapping
image_files = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.nrrd')])
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nrrd')])
data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

# 3. MONAI Transforms (Preprocessing)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]), # Naya method (Error fix)
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")), 
    ScaleIntensityRanged(
        keys=["image"], a_min=-1000, a_max=400,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"), 
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1, neg=1,
        num_samples=4, 
        image_key="image",
    ),
    ToTensord(keys=["image", "label"]),
])

# 4. Data Split (70% Train, 30% Val)
# 1. Split Numbers (91 total)
train_size = int(0.7 * len(data_dicts)) # 63
val_size = int(0.15 * len(data_dicts))  # 13-14
test_size = len(data_dicts) - train_size - val_size # 14

# 2. Files divide karna
train_files = data_dicts[:train_size]
val_files = data_dicts[train_size : train_size + val_size]
test_files = data_dicts[train_size + val_size :]

# 3. Datasets
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=train_transforms)
test_ds = Dataset(data=test_files, transform=train_transforms)

# 4. Loaders
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)
test_loader = DataLoader(test_ds, batch_size=1)

print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}") 

import torch
from monai.networks.nets import AttentionUnet
from monai.losses import DiceCELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Model Define karna (8GB GPU ke liye optimized)
model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1, # Segmentation ke liye
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

# 2. Loss Function (Medical Imaging ke liye best: Dice + CrossEntropy)
loss_function = DiceCELoss(sigmoid=True)

# 3. Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Step 5 Complete: Model {device} par load ho gaya hai!")

# --- Training Loop with Best Saving ---
max_epochs = 200
val_interval = 2 # Har 2 epochs baad check karega
best_metric = -1
best_metric_epoch = -1

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss/len(train_loader):.4f}")

   # Best Model Save karne ki logic
    if (epoch + 1) % val_interval == 0:
        current_loss = epoch_loss / len(train_loader)
        # Agar ye pehla epoch hai ya abhi tak ka sab se kam loss hai
        if best_metric == -1 or current_loss < best_metric:
            best_metric = current_loss
            torch.save(model.state_dict(), "best_simple_model.pth")
            print(f"New Best Model Saved! Loss: {best_metric:.4f}")