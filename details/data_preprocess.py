import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split
import random

# ---------- GLOBAL SEED (FIX 1) ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- 1. SET YOUR PATHS ----------
img_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\images\images"
mask_dir = r"D:\PE_Project\Dataset And Files\IEEE dataset\rs\rs"
output_base = r"D:\PE_Project\Preprocessed_Dataset_Segmentation"

# ---------- 2. CREATE FOLDERS ----------
for sub_folder in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base, sub_folder), exist_ok=True)

# ---------- 3. HELPER FUNCTION ----------
def process_single_case(f_name):
    img_path = os.path.join(img_dir, f_name)

    clean_name = f_name.replace('.nrrd', '')
    formatted_name = clean_name.zfill(4) if clean_name.isdigit() else clean_name
    mask_filename = f"{formatted_name}RefStd.nrrd"
    mask_path = os.path.join(mask_dir, mask_filename)

    if not os.path.isfile(mask_path):
        mask_path = os.path.join(mask_dir, f_name)
        if not os.path.isfile(mask_path):
            print(f"Skipping {f_name}")
            return None, None

    # Load
    img_obj = sitk.ReadImage(img_path)
    mask_obj = sitk.ReadImage(mask_path)

    img = sitk.GetArrayFromImage(img_obj).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_obj).astype(np.float32)

    spacing = img_obj.GetSpacing()

    # HU window
    img = np.clip(img, -1000, 400)

    # resample
    resize_factor = np.array(spacing[::-1]) / np.array([1, 1, 1])
    new_shape = np.round(img.shape * resize_factor)
    real_resize = new_shape / img.shape

    img = ndi.zoom(img, real_resize, order=1)
    mask = (ndi.zoom(mask, real_resize, order=0) > 0.5).astype(np.float32)

    # noise reduction
    img = ndi.gaussian_filter(img, sigma=0.5)

    # ROI + normalization
    roi = (img < -300).astype(np.uint8)
    pixels = img[roi > 0]

    if len(pixels) > 0:
        img = (img - np.mean(pixels)) / (np.std(pixels) + 1e-8)

    img[roi == 0] = -5

    # patch extraction
    size, stride = 64, 32
    pos_x, pos_y = [], []
    neg_x, neg_y = [], []

    threshold = 20

    D, H, W = img.shape

    for z in range(0, D-size, stride):
        for y in range(0, H-size, stride):
            for x in range(0, W-size, stride):

                p_img = img[z:z+size, y:y+size, x:x+size]
                p_mask = mask[z:z+size, y:y+size, x:x+size]

                if np.sum(p_mask) >= threshold:
                    pos_x.append(p_img)
                    pos_y.append(p_mask)
                else:
                    neg_x.append(p_img)
                    neg_y.append(p_mask)

    if len(pos_x) == 0:
        return None, None

    # balance 1:3
    neg_indices = list(range(len(neg_x)))
    random.shuffle(neg_indices)

    selected = neg_indices[:len(pos_x)*3]

    final_x = pos_x + [neg_x[i] for i in selected]
    final_y = pos_y + [neg_y[i] for i in selected]

    return np.array(final_x), np.array(final_y)

# ---------- 4. SPLIT (FIX 2: NO BALANCING HERE) ----------
all_files = [f for f in os.listdir(img_dir) if f.endswith('.nrrd')]

train_f, temp_f = train_test_split(all_files, test_size=0.3, random_state=SEED)
val_f, test_f = train_test_split(temp_f, test_size=0.5, random_state=SEED)

splits = {
    "train": train_f,
    "val": val_f,
    "test": test_f
}

# ---------- 5. EXECUTION ----------
for mode, file_list in splits.items():
    print(f"\n--- {mode.upper()} ---")

    for f in file_list:
        X, y = process_single_case(f)

        if X is not None:
            np.save(os.path.join(output_base, mode, f"{f}_X.npy"), X)
            np.save(os.path.join(output_base, mode, f"{f}_y.npy"), y)

            print(f"Saved {f} | patches: {len(X)}")

print("\nDONE ✔")