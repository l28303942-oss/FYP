#safe coding with 10 best models saving
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

# =========================
# 1. DEVICE & SEED
# =========================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. DATASET (As per original logic)
# =========================
class PEBalancedDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if "_X.npy" in f]
        self.index_map = []
        self.weights = []
        print(f"Indexing {folder}...")
        for f in self.files:
            y_arr = np.load(os.path.join(folder, f.replace("_X.npy", "_y.npy")), mmap_mode='r')
            for i in range(y_arr.shape[0]):
                self.index_map.append((f, i))
                is_positive = np.sum(y_arr[i]) > 20
                self.weights.append(10.0 if is_positive else 1.0)

    def __len__(self): return len(self.index_map)

    def augment(self, x, y):
        if torch.rand(1) > 0.5:
            d = torch.randint(1, 4, (1,)).item()
            x = torch.flip(x, dims=[d]); y = torch.flip(y, dims=[d])
        if torch.rand(1) > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[2, 3]); y = torch.rot90(y, k, dims=[2, 3])
        return x, y

    def __getitem__(self, idx):
        f, i = self.index_map[idx]
        x = np.load(os.path.join(self.folder, f), mmap_mode='r')[i]
        y = np.load(os.path.join(self.folder, f.replace("_X.npy", "_y.npy")), mmap_mode='r')[i]
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)
        x, y = self.augment(x, y)
        return x, y

# =========================
# 3. ARCHITECTURE (TS-Attention UNet)
# =========================
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

# =========================
# 4. LOSS & SCHEDULER
# =========================
class DiceFocalLoss(nn.Module):
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(); dice = 1 - (2. * inter + 1e-6) / (prob.sum() + target.sum() + 1e-6)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce); focal = (0.25 * (1-pt)**2 * bce).mean()
        return dice + focal

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, target_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs; self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.target_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.base_lrs]
        return self.base_lrs

# =========================
# 5. MAIN TRAINING
# =========================
def train():
    train_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\train"
    val_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\val"

    train_ds = PEBalancedDataset(train_path)
    sampler = WeightedRandomSampler(train_ds.weights, num_samples=len(train_ds), replacement=True)
    
    # Batch Size 4, Patience 70
    train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PEBalancedDataset(val_path), batch_size=4, shuffle=False)

    model = TS_Attention_UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 100; warmup_epochs = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs)
    warmup_sched = LinearWarmupScheduler(optimizer, warmup_epochs, 1e-4)
    criterion = DiceFocalLoss(); scaler = torch.amp.GradScaler("cuda")
    
    best_dice = 0; patience = 70; counter = 0
    accumulation_steps = 4 
    saved_models = [] # Top-K Buffer

    for epoch in range(epochs):
        model.train(); epoch_loss = 0; optimizer.zero_grad()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y) / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            epoch_loss += loss.item()

        if epoch < warmup_epochs: warmup_sched.step()
        else: scheduler.step()

        model.eval(); val_dice = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = (torch.sigmoid(model(x)) > 0.5).float()
                val_dice += (2 * (p*y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6)
        
        avg_dice = val_dice.item() / len(val_loader)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Dice: {avg_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.6e}")

        # --- SMART TOP-10 SAVING STRATEGY ---
        if avg_dice > best_dice:
            best_dice = avg_dice; counter = 0 # Reset Early Stopping
            save_name = f"model_dice_{avg_dice:.4f}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_name)
            saved_models.append((avg_dice, save_name))
            print(f"⭐ New Best Model Saved: {save_name}")

            # Keep only Top 10, remove the rest from disk
            if len(saved_models) > 10:
                saved_models.sort() # Sort by dice score
                worst_score, worst_file = saved_models.pop(0)
                if os.path.exists(worst_file):
                    os.remove(worst_file)
                    print(f"🗑️ Noise Removed: Deleted low-score model {worst_file}")
        else:
            counter += 1
            if counter >= patience:
                print(f"🛑 Stopping: No improvement for {patience} epochs.")
                break

if __name__ == "__main__":
    train()