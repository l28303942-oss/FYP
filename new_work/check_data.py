import os
import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset
from monai.utils import first  
import matplotlib.pyplot as plt # Ye repeat hai pr khair hai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, ToTensord
)
# LINE 11 YAHAN SE KHATAM KAR DEN

# 1. Paths (Wahi jo train script mein thay)
IMG_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
MASK_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

image_files = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.nrrd')])
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nrrd')])
data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

# 2. Transforms (Wahi purane)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")), 
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"), 
    ToTensord(keys=["image", "label"]),
])

# 3. Aik Sample uthayein aur Plot karein
check_ds = Dataset(data=data_dicts, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)

image, label = (check_data["image"][0][0], check_data["label"][0][0])
slice_idx = image.shape[2] // 2 

plt.figure("Check Data", (12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Image Slice {slice_idx}")
plt.imshow(image[:, :, slice_idx].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Label Slice")
plt.imshow(label[:, :, slice_idx].detach().cpu())
plt.show()