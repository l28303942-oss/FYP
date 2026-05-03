import os
import numpy as np

def check_patient(folder):
    for f in os.listdir(folder):
        if "_y.npy" in f:
            y = np.load(os.path.join(folder, f))

            pe = np.sum(y == 1)
            non = np.sum(y == 0)

            print(f"{f} -> PE: {pe}, NON-PE: {non}")

check_patient(r"D:\PE_Project\Preprocessed_Dataset_Segmentation\train")