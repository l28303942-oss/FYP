import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ARCHITECTURE (Aapki TS-Attention UNet)
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
# 2. DATASET (Preprocessing ke mutabiq)
# ==========================================
class PETestDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if "_X.npy" in f]
        self.index_map = []
        for f in self.files:
            x_arr = np.load(os.path.join(folder, f), mmap_mode='r')
            for i in range(x_arr.shape[0]):
                self.index_map.append((f, i))
    def __len__(self): return len(self.index_map)
    def __getitem__(self, idx):
        f, i = self.index_map[idx]
        x = np.load(os.path.join(self.folder, f), mmap_mode='r')[i].copy()
        y = np.load(os.path.join(self.folder, f.replace("_X.npy", "_y.npy")), mmap_mode='r')[i].copy()
        # Normalization wahi jo training mein thi
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        return torch.from_numpy(x).float().unsqueeze(0), torch.from_numpy(y).float().unsqueeze(0)

# ==========================================
# 3. SMART VISUALIZATION (Saboot Filter)
# ==========================================
def find_and_show_clots(model, loader, device, num_to_show=5):
    model.eval()
    found = 0
    with torch.no_grad():
        for x, y in loader:
            if found >= num_to_show: break
            
            # Prediction
            x_dev = x.to(device)
            # Threshold thora kam (0.3) kiya hai taake chote clots bhi pakray jaein
            output = torch.sigmoid(model(x_dev))
            pred = (output > 0.3).float().cpu()

            # --- SMART SLICE SCANNER ---
            # Hum poore 64 slices mein se wo slice dhundenge jahan model ne PE kaha hai
            best_slice = -1
            max_pixels = 0
            for s in range(64):
                pixels = pred[0, 0, s].sum().item()
                if pixels > max_pixels:
                    max_pixels = pixels
                    best_slice = s

            # Agar model ko clot mila hai (Ya asli mask mein clot hai)
            if best_slice != -1 or y.sum() > 20:
                if best_slice == -1: best_slice = 32 # Fallback
                
                img_slice = x[0, 0, best_slice].numpy()
                mask_slice = y[0, 0, best_slice].numpy()
                pred_slice = pred[0, 0, best_slice].numpy()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("CT Slice")
                plt.imshow(img_slice, cmap='gray')
                
                plt.subplot(1, 3, 2)
                plt.title("Doctor's Mask (Red)")
                plt.imshow(img_slice, cmap='gray')
                plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='Reds', alpha=0.6)
                
                plt.subplot(1, 3, 3)
                plt.title("Model Prediction (Green)")
                plt.imshow(img_slice, cmap='gray')
                plt.imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap='Greens', alpha=0.6)
                
                plt.show()
                found += 1

# ==========================================
# 4. RUN
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS_Attention_UNet().to(device)
    model.load_state_dict(torch.load('model_dice_0.7238_epoch_71.pth'))
    
    test_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\test"
    test_loader = DataLoader(PETestDataset(test_path), batch_size=1, shuffle=True)
    
    print("Clots scan ho rahe hain... Please wait.")
    find_and_show_clots(model, test_loader, device)