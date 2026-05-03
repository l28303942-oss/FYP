import numpy as np
import matplotlib.pyplot as plt
import os

# 1. File ka rasta (Path) set karein
folder_path = r"D:\PE_Project\Preprocessed_Dataset_Segmentation\train"
case_id = "003.nrrd" # Jis case ko dekhna chahte hain

# Files load karein
X = np.load(os.path.join(folder_path, f"{case_id}_X.npy"))
y = np.load(os.path.join(folder_path, f"{case_id}_y.npy"))

print(f"Total Patches in this file: {len(X)}")

# 2. Ek aisa patch select karein jis mein PE (Lesion) ho
# Hum pehla positive patch uthate hain
positive_indices = [i for i, patch in enumerate(y) if np.sum(patch) > 0]
if not positive_indices:
    print("Is file mein koi positive patch nahi mila!")
    idx = 0
else:
    idx = positive_indices[0] 

patch_img = X[idx]
patch_mask = y[idx]

# 3. Wo slice dhoondein jahan lesion sab se bara hai
slice_idx = np.argmax(np.sum(patch_mask, axis=(1, 2)))

# 4. Plotting
plt.figure(figsize=(15, 5))

# Original Image Slice
plt.subplot(1, 3, 1)
plt.imshow(patch_img[slice_idx], cmap='bone')
plt.title(f"CT Slice (Patch {idx}, Slice {slice_idx})")
plt.axis('off')

# Mask Slice
plt.subplot(1, 3, 2)
plt.imshow(patch_mask[slice_idx], cmap='Reds', alpha=0.8)
plt.title("PE Mask (Ground Truth)")
plt.axis('off')

# Overlay (Image + Mask)
plt.subplot(1, 3, 3)
plt.imshow(patch_img[slice_idx], cmap='bone')
plt.imshow(patch_mask[slice_idx], cmap='Reds', alpha=0.4) # Overlaying red mask
plt.title("Overlay (Lesion in Red)")
plt.axis('off')

plt.tight_layout()
plt.show()