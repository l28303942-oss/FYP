import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Dataset Class
class PEPatchDataset(Dataset):
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if "_X.npy" in f]
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_path = os.path.join(self.folder, self.files[idx])
        y_path = x_path.replace("_X.npy", "_y.npy")
        x = np.load(x_path)
        y = np.load(y_path)
        # Tensor conversion for visualization
        x = torch.from_numpy(x).float().unsqueeze(1)
        y = torch.from_numpy(y).float().unsqueeze(1)
        return x, y

# 2. Custom Collate (To handle different N)
def my_collate(batch):
    x = torch.cat([item[0] for item in batch], dim=0)
    y = torch.cat([item[1] for item in batch], dim=0)
    return x, y

# 3. Main Logic
if __name__ == "__main__":
    path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\train"
    ds = PEPatchDataset(path)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=my_collate)

    # Loader se data nikalna
    images, masks = next(iter(loader))
    print(f"Total Patches in this Batch: {images.shape[0]}")

    # Ek random patch select karein jis mein PE ho (mask sum > 0)
    found = False
    for i in range(images.shape[0]):
        if torch.sum(masks[i]) > 0:
            idx = i
            found = True
            break
    
    if not found: idx = 0 # Agar PE wala patch na milay toh pehla dikha do

    # Visualization
    slice_idx = 32 # 64 ka center slice
    img_slice = images[idx, 0, slice_idx].numpy()
    mask_slice = masks[idx, 0, slice_idx].numpy()

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title("CT Slice (Patch)")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_slice, cmap='Reds')
    plt.title("PE Mask (Ground Truth)")

    plt.subplot(1, 3, 3)
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(mask_slice, cmap='Reds', alpha=0.5) # Overlay
    plt.title("Overlay Check")

    plt.show()