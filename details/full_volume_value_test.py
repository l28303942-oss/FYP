import os
import numpy as np
import SimpleITK as sitk

def calculate_metrics():
    prediction_dir = r"D:\PE_Project\Inference_Results" 
    ground_truth_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs" 
    
    dice_scores = []
    iou_scores = []
    
    pred_files = [f for f in os.listdir(prediction_dir) if f.startswith("GREEN_MASK_")]
    
    print(f"Total {len(pred_files)} cases match ho rahe hain...\n")
    
    for p_file in pred_files:
        # 1. Filename formatting
        raw_name = p_file.replace("GREEN_MASK_", "").replace(".nrrd", "")
        
        if raw_name.isdigit():
            formatted_name = f"{int(raw_name):04d}RefStd.nrrd"
        else:
            formatted_name = f"{raw_name}RefStd.nrrd"
            
        gt_path = os.path.join(ground_truth_dir, formatted_name)
        
        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Asli mask nahi mila: {formatted_name}")
            continue
            
        # 2. Loading images
        pred_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(prediction_dir, p_file)))
        gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        
        # 3. Binary conversion
        pred_mask = (pred_img > 0.5).astype(np.float32)
        gt_mask = (gt_img > 0.5).astype(np.float32)
        
        # ==========================================
        # 🔍 DIAGNOSTIC SECTION (Naya Hisa)
        # ==========================================
        p_sum = np.sum(pred_mask)
        g_sum = np.sum(gt_mask)
        p_max = np.max(pred_mask)
        g_max = np.max(gt_mask)
        
        print(f"\n🔍 Case: {formatted_name}")
        print(f"   - Prediction: Max Val={p_max}, Total Pixels={p_sum}")
        print(f"   - Ground Truth: Max Val={g_max}, Total Pixels={g_sum}")
        # ==========================================

        # 4. Metrics Calculation
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        union_area = np.sum((pred_mask + gt_mask) > 0)
        iou = (intersection + 1e-7) / (union_area + 1e-7)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        print(f"   ✅ Dice Score: {dice:.4f}")

    if dice_scores:
        print("\n" + "="*35)
        print(f"FINAL AVERAGE DICE: {np.mean(dice_scores):.4f}")
        print(f"FINAL AVERAGE IoU:  {np.mean(iou_scores):.4f}")
        print("="*35)
    else:
        print("❌ Error: Koi matching file nahi mili!")

if __name__ == "__main__":
    calculate_metrics()