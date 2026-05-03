# Pulmonary Embolism — Deep Learning Research Pipeline

End-to-end **segmentation** (Attention Residual U-Net 3D) + **classification** (EfficientNet-B0) with **patient-wise splits**, **lung-region foreground cropping** (via MONAI `CropForeground`), class-imbalance-aware losses, AMP, and evaluation reports.

## Dataset layout (default paths)

Set in `config.py` or environment variables `PE_IMG_DIR` / `PE_MASK_DIR`:

- Images: `D:\my_project\Dataset And Files\IEEE dataset\images\images\*.nrrd`
- Masks: `D:\my_project\Dataset And Files\IEEE dataset\rs\rs\*.nrrd`

**Pairing rule:** same filename stem for image and mask.

## Setup

```powershell
cd D:\my_project\pe_dl_research
python -m pip install -r requirements_dl.txt
```

Run all scripts from **`pe_dl_research`** so imports resolve.

### GPU (8 GB NVIDIA recommended)

Training **uses CUDA automatically** when available (`torch.cuda.is_available()`).

1. Install a **CUDA-enabled** PyTorch (not the CPU-only wheel). See **`INSTALL_GPU.md`**.
2. Verify:

   ```powershell
   python check_gpu.py
   ```

3. Scripts print `[device] GPU 0: ... ~X GiB` at startup.

### All work on **D:** drive (recommended on your PC)

By default **checkpoints, splits, figures** go to **`D:\my_project\pe_dl_outputs`** (not C:).

One command sets **TEMP**, **caches**, **outputs** on **D:** and runs training on **GPU**:

```powershell
cd D:\my_project\pe_dl_research
.\run_training_gpu_D.ps1
```

Requires **CUDA PyTorch** first (`.\install_cuda_pytorch.ps1` if `check_gpu.py` shows CPU).

Optional env vars for **VRAM (~8 GB)**:

- `PE_SEG_BATCH` — segmentation batch size (default `2`; try `1` if OOM)
- `PE_SW_BATCH` — sliding-window batch during validation (default `2`)
- `PE_AMP` — `1` (default) mixed precision on GPU; `0` to disable

## Pipeline order

1. **Analyze dataset** (optional but recommended)

   ```powershell
   python dataset_analysis.py
   ```

   Writes `outputs/reports/dataset_analysis.json`.

2. **Build patient-wise split manifest**

   ```powershell
   python preprocessing.py
   ```

   Writes `outputs/splits/patientwise_split.json`.  
   **Patient IDs** are inferred from filename stems (`utils.extract_patient_id`). Adjust heuristics if your naming differs.

3. **Train segmentation**

   ```powershell
   python train_segmentation.py
   ```

   Saves `outputs/checkpoints/best_seg_attention_unet3d.pth` and `outputs/figures/seg_training_curves.png`.

4. **Train classification**

   ```powershell
   python train_classification.py
   ```

   Saves `outputs/checkpoints/best_cls_efficientnet.pth`.

5. **Evaluate on test split**

   ```powershell
   python evaluation.py
   ```

   Writes `outputs/reports/evaluation_report.json` and confusion matrix image.

6. **Inference on new volume**

   ```powershell
   python inference.py --input path\to\ct.nrrd
   ```

   Outputs probability map + binary mask NRRD under `outputs/predictions/`.

## Methods summary

| Piece | Implementation |
|--------|----------------|
| Split | **Patient-grouped** train/val/test (no slice leakage across patients) |
| Lung ROI | **HU foreground** + `CropForegroundd` (same as typical lung CT pipeline) |
| Segmentation | **Attention gates + residual blocks**, Dice+Focal, AdamW, cosine LR, early stopping, AMP |
| Classification | **EfficientNet-B0**, slice labels from masks, weighted sampling stub (extend with precomputed slice weights) |
| Metrics | Dice, IoU, precision, recall (seg); accuracy, F1, confusion matrix, ROC/AUC (cls) |

## Generalization (beyond this dataset)

Training uses **HU random jitter** and **noise** on classification slices; segmentation uses **intensity augmentations**. For **external hospitals**, expect domain shift: retrain/fine-tune with a small amount of target-domain data, align **slice thickness / kernel**, and standardize **resampling** (see MONAI `Spacingd`). Inference assumes similar **CT lung** statistics; non-CT modalities require a different pipeline.

## Cross-validation

Not enabled by default (long runs). You can add K-fold over **patient IDs** by splitting `group_pairs_by_patient` in `preprocessing.py` into K folds.

## Quality checklist if metrics are weak

- Verify mask alignment and **same shape** as image for every pair.
- Check **class imbalance** (very sparse PE): increase **focal** weight in `config.py`, adjust `RandCropByPosNegLabeld` pos/neg counts.
- Reduce **false positives**: raise `SEG_PRED_THRESHOLD`, add **FP-focused** post-processing, or small **CRF** (future work).

## Files

| Module | Role |
|--------|------|
| `config.py` | Paths, hyperparameters |
| `utils.py` | Seeds, patient ID parsing |
| `dataset_analysis.py` | Structure / corruption report |
| `preprocessing.py` | Pairing, splits, MONAI transforms |
| `models.py` | Attention Residual U-Net 3D, EfficientNet classifier |
| `cls_data.py` | Classification slice dataset |
| `train_segmentation.py` | Seg training |
| `train_classification.py` | Cls training |
| `evaluation.py` | Test metrics |
| `inference.py` | Deploy inference |
