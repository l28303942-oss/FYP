import os

pred_dir = r"D:\PE_Project\Inference_Results"
gt_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"

print("--- Prediction Folder ki pehli 3 files ---")
print(os.listdir(pred_dir)[:3])

print("\n--- Original (rs/rs) Folder ki pehli 3 files ---")
print(os.listdir(gt_dir)[:3])