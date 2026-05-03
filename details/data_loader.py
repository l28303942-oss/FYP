import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class PEDataset(Dataset):
    def __init__(self, base_path, mode='train'):
        self.mode_path = os.path.join(base_path, mode)
        # Check if path exists
        if not os.path.exists(self.mode_path):
            raise FileNotFoundError(f"Path not found: {self.mode_path}")
        self.file_names = [f for f in os.listdir(self.mode_path) if f.endswith('_X.npy')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        mask_name = img_name.replace('_X.npy', '_y.npy')
        
        image = np.load(os.path.join(self.mode_path, img_name))
        mask = np.load(os.path.join(self.mode_path, mask_name))
        
        # [N, 64, 64, 64] -> [N, 1, 64, 64, 64]
        image = torch.from_numpy(image).float().unsqueeze(1)
        mask = torch.from_numpy(mask).float().unsqueeze(1)
        
        return image, mask

# --- THE FIX ---
def my_collate_fn(batch):
    # Batch is a list of tuples (image, mask)
    # We concatenate them along the first dimension (N)
    images = torch.cat([item[0] for item in batch], dim=0)
    masks = torch.cat([item[1] for item in batch], dim=0)
    return images, masks

# ---------- SETUP ----------
base_dir = r"D:\PE_Project\Preprocessed_Dataset_Segmentation"

train_ds = PEDataset(base_dir, mode='train')

# Yahan dekhein: collate_fn=my_collate_fn lazmi hona chahiye
train_loader = DataLoader(
    train_ds, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=my_collate_fn  # <--- YE LINE SAB SE ZAROORI HAI
)

# ---------- TEST ----------
print(f"Total batches: {len(train_loader)}")

try:
    print("\n--- Testing Loader ---")
    # Get one batch
    data_iter = iter(train_loader)
    images, masks = next(data_iter)
    
    print("✅ MUBARAK HO! Data load ho gaya.")
    print(f"Final Batch Shape: {images.shape}")
    print(f"Mask Shape: {masks.shape}")

except Exception as e:
    print(f"❌ Abhi bhi masla hai: {e}")