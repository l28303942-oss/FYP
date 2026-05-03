# Training on your NVIDIA GPU (8 GB)

The code **automatically uses `cuda:0`** when `torch.cuda.is_available()` is `True`.

## 1) Check

```powershell
cd D:\my_project\pe_dl_research
python check_gpu.py
```

You want: `cuda.is_available() True` and your GPU name.

## 2) If CUDA is `False` (CPU-only PyTorch)

Uninstall the CPU build and install the **CUDA** build that matches your driver. Example for **CUDA 12.4** wheels:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Disk full (`No space left on device`) during install

The CUDA wheel is **~2.5 GB**. Pip unpacks under **`%TEMP%`** (often on **C:**). If C: is full:

1. Free **at least 4 GB** on **C:**, **or**
2. Point temp to another drive (e.g. **D:**):

   ```powershell
   mkdir D:\pip_temp 2>nul
   $env:TEMP="D:\pip_temp"
   $env:TMP="D:\pip_temp"
   pip cache purge
   pip uninstall torch torchvision -y
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

3. Run `python check_gpu.py` — you should see **+cu124** in the torch version and `cuda.is_available() True`.

Other options: [pytorch.org](https://pytorch.org) → select Windows + Pip + your CUDA version (`cu121`, `cu118`, etc.).

Update GPU driver if install fails or GPU is not listed in `nvidia-smi`.

## 3) 8 GB VRAM tips (reduce OOM)

Set **before** training (PowerShell):

```powershell
$env:PE_SEG_BATCH="1"          # smaller 3D batch
$env:PE_SW_BATCH="1"            # smaller sliding-window batch in validation
$env:PE_PATCH="80,80,80"       # smaller patches (must edit preprocessing consistency)
```

Defaults (`SEG_BATCH=2`, AMP on GPU) are safe on many 8 GB cards for **96³** patches.

## 4) Force CPU only (debug)

```powershell
$env:PE_FORCE_CPU="1"
python train_segmentation.py
```

## 5) Fail if GPU missing (avoid slow CPU by mistake)

```powershell
$env:PE_REQUIRE_GPU="1"
python train_segmentation.py
```
