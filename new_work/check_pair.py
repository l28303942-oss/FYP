import os
from glob import glob

# 1. Paths set karein
IMG_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
MASK_DIR = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

# 2. Files ki list banayein (Sorted taake image aur mask match karein)
# Hum filenames ko sort karenge taake 001.nrrd ke sath 0001RefStd.nrrd hi aaye
image_files = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.nrrd')])
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nrrd')])

# 3. Data Dictionary banayein
data_dicts = [{"image": img, "label": mask} for img, mask in zip(image_files, mask_files)]

# Check karein ke pehla pair sahi hai ya nahi
print(f"Total pairs found: {len(data_dicts)}")
print(f"First Pair:\nImage: {data_dicts[0]['image']}\nMask: {data_dicts[0]['label']}")