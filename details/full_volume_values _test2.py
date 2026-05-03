import os
import numpy as np
import SimpleITK as sitk

def calculate_metrics(pred_mask, gt_mask):
    # Intersection aur Union nikaalne ke liye
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    union = np.sum((pred_mask == 1) | (gt_mask == 1))
    
    # Dice Score
    volume_sum = np.sum(pred_mask) + np.sum(gt_mask)
    dice = (2. * intersection) / volume_sum if volume_sum > 0 else 1.0
    
    # IoU Score
    iou = intersection / union if union > 0 else 1.0
    
    return dice, iou

# [✓] PATHS
pred_dir = r"D:\PE_Project\Inference_Results"
gt_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

all_dice = []
all_iou = []
files = [f for f in os.listdir(pred_dir) if f.startswith("GREEN_MASK_")]

print(f"📊 Total {len(files)} files ka evaluation shuru ho raha hai...\n")

for f in files:
    pred_path = os.path.join(pred_dir, f)
    pred_img = sitk.ReadImage(pred_path)
    pred_arr = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)

    original_name = f.replace("GREEN_MASK_", "")
    
    # Ground Truth file matching
    gt_path = os.path.join(gt_dir, original_name)
    if not os.path.exists(gt_path):
        clean_name = original_name.replace('.nrrd', '')
        formatted_name = clean_name.zfill(4) if clean_name.isdigit() else clean_name
        gt_path = os.path.join(gt_dir, f"{formatted_name}RefStd.nrrd")

    if os.path.exists(gt_path):
        gt_img = sitk.ReadImage(gt_path)
        gt_arr = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)

        # Metrics calculate karein
        dice, iou = calculate_metrics(pred_arr, gt_arr)
        
        all_dice.append(dice)
        all_iou.append(iou)
        print(f"📄 File: {original_name} | Dice: {dice:.4f} | IoU: {iou:.4f}")
    else:
        print(f"⚠️ Warning: Ground Truth missing for {original_name}")

if len(all_dice) > 0:
    print("\n" + "="*40)
    print(f"🏆 FINAL AVERAGE RESULTS:")
    print(f"Mean Dice Score: {np.mean(all_dice):.4f}")
    print(f"Mean IoU Score:  {np.mean(all_iou):.4f}")
    print("="*40)
else:
    print("❌ Koi result nahi mila. Please paths check karein.")