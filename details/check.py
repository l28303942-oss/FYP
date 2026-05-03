import numpy as np
import os

def count_labels(folder):
    total_pe = 0
    total_non_pe = 0

    for f in os.listdir(folder):
        if f.endswith("_y.npy"):
            y = np.load(os.path.join(folder, f))

            total_pe += np.sum(y == 1)
            total_non_pe += np.sum(y == 0)

    return total_pe, total_non_pe


train_pe, train_non = count_labels(r"D:\PE_Project\Preprocessed_Dataset_Segmentation\train")

print("TRAIN PE:", train_pe)
print("TRAIN NON-PE:", train_non)
print("Ratio:", round(train_non / (train_pe + 1e-6), 2))