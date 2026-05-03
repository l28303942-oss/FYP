import os
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split

# ==========================================
# 1. ARCHITECTURE (TS-Attention UNet)
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
# 2. FINAL INFERENCE LOGIC (SAFE VERSION)
# ==========================================
def run_final_inference():
    SEED = 42
    # [✓] PATHS
    img_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
    output_dir = r"D:\PE_Project\Inference_Results"
    # UPDATED WEIGHT PATH (Epoch 68 is the last stable one)
    checkpoint_path = "model_dice_0.6834_epoch_68.pth" 
    
    os.makedirs(output_dir, exist_ok=True)

    # [✓] TEST SELECTION
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nrrd')])
    _, temp_f = train_test_split(all_files, test_size=0.3, random_state=SEED)
    _, test_f = train_test_split(temp_f, test_size=0.5, random_state=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS_Attention_UNet().to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"✅ STABLE MODEL LOADED: {checkpoint_path}")
    else:
        print(f"❌ ERROR: Checkpoint '{checkpoint_path}' nahi mila. Check karein ke ye file folder mein hai.")
        return

    for f_name in test_f:
        print(f"🔄 Processing: {f_name}...")
        img_path = os.path.join(img_dir, f_name)
        img_obj = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_obj).astype(np.float32)
        spacing = img_obj.GetSpacing()

        # [1] PRE-PROCESSING (As per stable training)
        img_proc = np.clip(img_array, -1000, 400)
        
        # Resampling
        rf = np.array(spacing[::-1]) / np.array([1, 1, 1])
        new_shape = np.round(img_proc.shape * rf)
        real_resize = new_shape / img_proc.shape
        img_proc = ndi.zoom(img_proc, real_resize, order=1)
        
        # [2] SAFE NORMALIZATION
        roi = (img_proc < -300).astype(np.uint8)
        pixels = img_proc[roi > 0]
        
        if len(pixels) > 10 and np.std(pixels) > 1e-5:
            img_proc = (img_proc - np.mean(pixels)) / (np.std(pixels) + 1e-8)
        else:
            img_proc = (img_proc - np.mean(img_proc)) / (np.std(img_proc) + 1e-8)
        
        img_proc[roi == 0] = -5
        img_proc = np.nan_to_num(img_proc)

        # [3] PATCH-BASED INFERENCE (Stitching)
        D, H, W = img_proc.shape
        full_pred = np.zeros_like(img_proc)
        count_map = np.zeros_like(img_proc)
        size, stride = 64, 32

        with torch.no_grad():
            for z in range(0, D-size+1, stride):
                for y in range(0, H-size+1, stride):
                    for x in range(0, W-size+1, stride):
                        patch = img_proc[z:z+size, y:y+size, x:x+size]
                        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                        
                        output = torch.sigmoid(model(patch_t))
                        full_pred[z:z+size, y:y+size, x:x+size] += output.cpu().numpy()[0, 0]
                        count_map[z:z+size, y:y+size, x:x+size] += 1

       # [4] POST-PROCESSING (REPLACE THIS SECTION)
        avg_prob = np.divide(full_pred, count_map, out=np.zeros_like(full_pred), where=count_map!=0)
        avg_prob = np.nan_to_num(avg_prob)
        
        # 1. Threshold ko 0.5 karein (Standard for Dice)
        mask_proc = (avg_prob > 0.5).astype(np.float32)

        # 2. Small Noise Removal (Connected Components)
        # Agar chote chote dots hain toh unhein khatam karne ke liye:
        from skimage import morphology
        mask_bool = mask_proc > 0.5
        mask_cleaned = morphology.remove_small_objects(mask_bool, min_size=50) 
        mask_proc = mask_cleaned.astype(np.float32)

        # 3. Resample back to original scan shape
        final_mask_full = (ndi.zoom(mask_proc, 1/real_resize, order=0) > 0.5).astype(np.uint8)

        # [5] SAVE RESULT
        res_obj = sitk.GetImageFromArray(final_mask_full)
        res_obj.CopyInformation(img_obj)
        save_path = os.path.join(output_dir, f"GREEN_MASK_{f_name}")
        sitk.WriteImage(res_obj, save_path)
        
        print(f"✨ SUCCESS: {f_name} | Max Prob: {np.max(avg_prob):.4f} | Sum: {np.sum(final_mask_full)}")

if __name__ == "__main__":
    run_final_inference()