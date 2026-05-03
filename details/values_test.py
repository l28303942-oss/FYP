import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ARCHITECTURE (Aapki train_v5 wali classes)
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
# 2. DATASET CLASS (Simplified for Testing)
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
        
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)
        return x, y

# ==========================================
# 3. TEST FUNCTION
# ==========================================
def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Initialize karein (Architecture name fixed)
    model = TS_Attention_UNet().to(device)
    
    # Best Model Load karein
    model_path = "model_dice_0.7238_epoch_71.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded: {model_path}")
    else:
        print(f"❌ Error: {model_path} nahi mili!")
        return

    # Data Path (Aapki screenshot ke mutabiq)
    test_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\test"
    test_loader = DataLoader(PETestDataset(test_path), batch_size=2, shuffle=False)

    model.eval()
    total_dice = 0
    total_iou = 0

    print("Testing shuru ho rahi hai...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            p = (prob > 0.5).float()

            # Dice & IoU
            inter = (p * y).sum()
            union = (p + y).sum() - inter
            
            dice = (2 * inter + 1e-6) / (p.sum() + y.sum() + 1e-6)
            iou = (inter + 1e-6) / (union + 1e-6)

            total_dice += dice.item()
            total_iou += iou.item()

            if (i + 1) % 10 == 0:
                print(f"Batch {i+1} done...")

    print("\n" + "="*30)
    print(f"FINAL RESULTS:")
    print(f"Test Dice: {total_dice / len(test_loader):.4f}")
    print(f"Test IoU:  {total_iou / len(test_loader):.4f}")
    print("="*30)

if __name__ == "__main__":
    run_test()