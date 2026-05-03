import os
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
from monai.inferers import sliding_window_inference
# Aapka TS-Attention UNet Architecture yahan lazmi hona chahiye
# (TAD, GTAFM, TS_Attention_UNet classes ko yahan paste karein)
#  3. ARCHITECTURE (TS-Attention UNet)
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

def run_final_inference():
    # [1] SETTINGS
    img_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
    output_dir = r"D:\PE_Project\Inference_Results"
    checkpoint_path = "model_dice_0.6834_epoch_68.pth"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS_Attention_UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Sirf un files par chalayein jin par error aa raha tha
    test_files = [f for f in os.listdir(img_dir) if f.endswith('.nrrd')]

    for f_name in test_files:
        print(f"🔄 Processing: {f_name}")
        img_obj = sitk.ReadImage(os.path.join(img_dir, f_name))
        img_array = sitk.GetArrayFromImage(img_obj).astype(np.float32)
        
        # [2] PRE-PROCESSING (Standardized)
        img_proc = np.clip(img_array, -1000, 400)
        
        # Normalization (Wahi jo training mein thi)
        mean, std = np.mean(img_proc), np.std(img_proc)
        img_proc = (img_proc - mean) / (std + 1e-8)

        # Tensor conversion (B, C, D, H, W)
        input_tensor = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).to(device)

        # [3] SLIDING WINDOW INFERENCE (No more manual loops!)
        with torch.no_grad():
            output = sliding_window_inference(
                inputs=input_tensor, 
                roi_size=(64, 64, 64), 
                sw_batch_size=4, 
                predictor=model,
                overlap=0.5,      # 50% overlap for better accuracy
                mode="gaussian"    # Smooths the edges between patches
            )
            output = torch.sigmoid(output).cpu().numpy()[0, 0]

        # [4] CLEANING & SAVING
        final_mask = (output > 0.5).astype(np.uint8) # Thresholding
        
        res_obj = sitk.GetImageFromArray(final_mask)
        res_obj.CopyInformation(img_obj) # Copy metadata (Origin, Spacing, Direction)
        
        save_path = os.path.join(output_dir, f"GREEN_MASK_{f_name}")
        sitk.WriteImage(res_obj, save_path)
        print(f"✅ Saved: {f_name}")

if __name__ == "__main__":
    run_final_inference()