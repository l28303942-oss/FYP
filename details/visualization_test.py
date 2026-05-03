import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ARCHITECTURE (Fixed and Ready)
# ==========================================
class TAD(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, c, 3, padding=1); self.sig = nn.Sigmoid()
    def forward(self, x): return x * self.sig(self.conv(x))

class GTAFM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(c, c, 3, padding=1), nn.ReLU(), nn.Conv3d(c, c, 3, padding=1))
    def forward(self, x): return x + self.net(x)

class TS_Attention_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(), TAD(32))
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), GTAFM(64))
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), GTAFM(128))
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2); self.att2 = TAD(64) 
        self.dec2 = nn.Sequential(nn.Conv3d(128, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2); self.att1 = TAD(32)
        self.dec1 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU())
        self.final = nn.Conv3d(32, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), self.att2(e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(e1)], dim=1))
        return self.final(d1)

# ==========================================
# 2. DATASET CLASS (Optimized for Testing)
# ==========================================
class PETestDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if "_X.npy" in f]
        self.index_map = []
        print(f"Dataset scan ho raha hai (sirf clots)... {folder}")
        for f in self.files:
            x_arr = np.load(os.path.join(folder, f), mmap_mode='r')
            y_path = os.path.join(folder, f.replace("_X.npy", "_y.npy"))
            
            if os.path.exists(y_path):
                # Poore volume mein PE hai ya nahi ye check karte hain
                y_volume = np.load(y_path, mmap_mode='r')
                if np.sum(y_volume) > 50: # Threshold taake noise skip ho jaye
                    for i in range(x_arr.shape[0]):
                        self.index_map.append((f, i))
        print(f"Total slices matched with clots: {len(self.index_map)}")

    def __len__(self): return len(self.index_map)
    def __getitem__(self, idx):
        f, i = self.index_map[idx]
        x = np.load(os.path.join(self.folder, f), mmap_mode='r')[i].copy()
        y = np.load(os.path.join(self.folder, f.replace("_X.npy", "_y.npy")), mmap_mode='r')[i].copy()
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)
        return x, y

# ==========================================
# 3. SABOOT / FILTER VISUALIZATION FUNCTION
# ==========================================
def show_clot_saboot(model, loader, device, num_samples=10):
    model.eval()
    count = 0
    with torch.no_grad():
        for x, y in loader:
            if count >= num_samples: break
            
            # --- AGGRESSIVE CLOT FILTER (NEW) ---
            if y.sum() < 20: continue # noise se bachne ke liye filter

            x_dev, y_dev = x.to(device), y.to(device)
            logits = model(x_dev)
            pred = (torch.sigmoid(logits) > 0.5).float()

            # --- Slicing strategy for best view ---
            slice_idx = x.shape[2] // 2
            # Middle slice agar khali ho to PE wali slice dhundte hain
            y_slice = y[0, 0].cpu().numpy()
            if np.sum(y_slice) < 5:
                potential_slices = np.argwhere(y[0, 0].cpu().numpy() > 0)
                if len(potential_slices) > 0:
                    slice_idx = potential_slices[len(potential_slices)//2][0]

            img = x[0, 0, slice_idx].cpu().numpy()
            mask = y[0, 0, slice_idx].cpu().numpy()
            prediction = pred[0, 0, slice_idx].cpu().numpy()

            # --- VISUALIZATION PLOT ---
            plt.figure(figsize=(18, 6))
            
            # Subplot 1: Original CT Scan
            plt.subplot(1, 3, 1)
            plt.title(f"Sample {count+1}: CT Scan (Middle Slice)")
            plt.imshow(img, cmap='gray')
            plt.axis('off')

            # Subplot 2: True Clot (Red Area)
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth (Clot Area - RED)")
            plt.imshow(mask, cmap='Reds', alpha=0.9)
            plt.axis('off')

            # Subplot 3: Model Prediction (Green Area)
            plt.subplot(1, 3, 3)
            plt.title("Model Prediction (Green Area)")
            plt.imshow(prediction, cmap='Greens', alpha=0.9)
            plt.axis('off')

            plt.show()
            count += 1
            print(f"Sample {count} with PE shown.")

# ==========================================
# 4. MAIN EXECUTION (Paths as per multan location)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # paths (correct as per multan environment)
    model_path = "model_dice_0.7238_epoch_71.pth"
    test_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\test"

    # Load Model
    model = TS_Attention_UNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded weights: {model_path}")
    else:
        print(f"❌ Error: Model weights not found at {model_path}")
        exit()

    # Load Data (Filter applied in Dataset class)
    # Shuffle=True, isliye har bar naya clot milega
    test_loader = DataLoader(PETestDataset(test_path), batch_size=1, shuffle=True) 

    # Saboot dekhne ka waqt
    print("Pre-scanning done. Generating 10 images with clots. Close window to see next.")
    show_clot_saboot(model, test_loader, device, num_samples=10)
    print("\nThank you for patience. All 10 clot samples generated.")