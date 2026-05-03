import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# DATASET
# =========================
class PEPatchDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if "_X.npy" in f]

        self.index_map = []
        for f in self.files:
            arr = np.load(os.path.join(folder, f), mmap_mode='r')
            for i in range(arr.shape[0]):
                self.index_map.append((f, i))

    def __len__(self):
        return len(self.index_map)

    def augment(self, x, y):
        if torch.rand(1) > 0.5:
            d = torch.randint(1, 4, (1,)).item()
            x = torch.flip(x, dims=[d])
            y = torch.flip(y, dims=[d])

        if torch.rand(1) > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[2, 3])
            y = torch.rot90(y, k, dims=[2, 3])

        if torch.rand(1) > 0.3:
            x = x + torch.randn_like(x) * 0.01

        return x, y

    def __getitem__(self, idx):
        f, i = self.index_map[idx]

        x = np.load(os.path.join(self.folder, f), mmap_mode='r')[i]
        y = np.load(os.path.join(self.folder, f.replace("_X.npy", "_y.npy")), mmap_mode='r')[i]

        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.from_numpy(y).float().unsqueeze(0)

        x, y = self.augment(x, y)
        return x, y

# =========================
# MODEL
# =========================
class TAD(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, c, 3, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.conv(x))


class GTAFM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(c, c, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)


class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            TAD(32)
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            GTAFM(64)
        )

        self.up = nn.ConvTranspose3d(64, 32, 2, stride=2)

        self.dec = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d = self.up(e2)

        x = torch.cat([d, e1], dim=1)
        return self.dec(x)

# =========================
# LOSS
# =========================
class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = torch.clamp(prob, 1e-6, 1 - 1e-6)

        tp = (prob * target).sum(dim=[1,2,3,4])
        fn = ((1 - prob) * target).sum(dim=[1,2,3,4])
        fp = (prob * (1 - target)).sum(dim=[1,2,3,4])

        tversky = (tp + 1e-6) / (tp + self.alpha*fn + self.beta*fp + 1e-6)
        tversky_loss = 1 - tversky.mean()

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        focal = ((1 - prob) ** self.gamma * bce).mean()

        return tversky_loss + focal

# =========================
# TRAINING
# =========================
def train():
    train_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\train"
    val_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\val"

    train_loader = DataLoader(
        PEPatchDataset(train_path),
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        PEPatchDataset(val_path),
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = HybridNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    criterion = TverskyFocalLoss()

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_dice = 0
    accumulation = 4
    epochs = 100

    # ✔ EARLY STOPPING ADDED
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = criterion(out, y) / accumulation
                scaler.scale(loss).backward()
            else:
                out = model(x)
                loss = criterion(out, y) / accumulation
                loss.backward()

            if (i + 1) % accumulation == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            loss_sum += loss.item()

        # ===== VALIDATION =====
        model.eval()
        dice_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                pred = torch.sigmoid(model(x))
                pred = (pred > 0.5).float()

                inter = (pred * y).sum(dim=[1,2,3,4])
                union = pred.sum(dim=[1,2,3,4]) + y.sum(dim=[1,2,3,4])

                dice = (2*inter + 1e-6) / (union + 1e-6)
                dice_total += dice.mean().item()

        avg_dice = dice_total / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1} | Loss: {loss_sum/len(train_loader):.4f} | Dice: {avg_dice:.4f}")

        # ✔ BEST MODEL + EARLY STOPPING LOGIC
        if avg_dice > best_dice:
            best_dice = avg_dice
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✔ Best Model Saved")
        else:
            counter += 1
            print(f"No improvement: {counter}/{patience}")

        if counter >= patience:
            print("⛔ Early Stopping Triggered")
            break

if __name__ == "__main__":
    train()