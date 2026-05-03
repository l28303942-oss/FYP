import os
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

# ==========================================
# [✓] STEP 1: ARCHITECTURE (Essential for loading weights)
# ==========================================
class TAD(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, c, 3, padding=1)
        self.sig = nn.Sigmoid()
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
# [✓] STEP 2: FINAL INFERENCE LOGIC
# ==========================================
def run_final_inference():
    SEED = 42
    img_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
    output_dir = r"D:\PE_Project\Inference_Results"
    os.makedirs(output_dir, exist_ok=True)

    # [✓] Strict Test Selection (Same as Pre-processing)
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nrrd')])
    _, temp_f = train_test_split(all_files, test_size=0.3, random_state=SEED)
    _, test_f = train_test_split(temp_f, test_size=0.5, random_state=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS_Attention_UNet().to(device)
    
    # [✓] Correct Model Path
    checkpoint_path = "model_dice_0.7238_epoch_71.pth" 
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"✅ FINAL MODEL LOADED: {checkpoint_path}")
    else:
        print(f"❌ ERROR: Model file '{checkpoint_path}' nahi mili. Check karein ke ye script ke saath pari hai.")
        return

    model.eval()

    for f_name in test_f:
        print(f"🔄 Processing: {f_name}")
        img_path = os.path.join(img_dir, f_name)
        img_obj = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_obj).astype(np.float32)

        # [✓] Essential Pre-processing
        img_proc = np.clip(img_array, -1000, 400)
        img_proc = (img_proc - np.mean(img_proc)) / (np.std(img_proc) + 1e-8)

        # [✓] Stitching / Reconstruction
        full_pred = np.zeros_like(img_array)
        count_map = np.zeros_like(img_array)
        size, stride = 64, 32

        with torch.no_grad():
            for z in range(0, img_array.shape[0]-size+1, stride):
                for y in range(0, img_array.shape[1]-size+1, stride):
                    for x in range(0, img_array.shape[2]-size+1, stride):
                        patch = img_proc[z:z+size, y:y+size, x:x+size]
                        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                        output = torch.sigmoid(model(patch_t))
                        pred_patch = (output > 0.5).float().cpu().numpy()[0, 0]
                        
                        full_pred[z:z+size, y:y+size, x:x+size] += pred_patch
                        count_map[z:z+size, y:y+size, x:x+size] += 1

        # final_mask = (np.where(count_map > 0, full_pred/count_map, 0) > 0.5).astype(np.uint8)
        # [✓] Line 100 ko is se replace karein (Warning fix karne ke liye)
        final_mask = (np.where(count_map > 0, full_pred / (count_map + 1e-8), 0) > 0.5).astype(np.uint8)

        # [✓] Metadata Copy (Alignment Guarantee)
        res_obj = sitk.GetImageFromArray(final_mask)
        res_obj.CopyInformation(img_obj) # <--- Ye line theek kar di hai
        
        save_path = os.path.join(output_dir, f"GREEN_MASK_{f_name}")
        sitk.WriteImage(res_obj, save_path)
        print(f"✨ Success: Saved {save_path}")

if __name__ == "__main__":
    run_final_inference()