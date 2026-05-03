# Training on GPU + all heavy paths on D: drive (TEMP, caches, outputs).
# Run from PowerShell:  cd D:\my_project\pe_dl_research ; .\run_training_gpu_D.ps1
# Prerequisites: CUDA PyTorch installed (see INSTALL_GPU.md). Run: python check_gpu.py
#
# Live log file for VS Code: pe_dl_research\train.txt (same folder as this script)

$ErrorActionPreference = "Stop"
$Root = "D:\my_project"
$TrainLog = Join-Path $Root "pe_dl_research\train.txt"

function Stop-TrainTranscript {
    try { Stop-Transcript -ErrorAction SilentlyContinue } catch {}
}

# New run: header + start transcript (if train.txt is open in VS Code, use a new session file)
$hdr = "=== PE DL training log started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
try {
    Set-Content -Path $TrainLog -Value $hdr -Encoding utf8 -ErrorAction Stop
} catch {
    $TrainLog = Join-Path $Root "pe_dl_research\train_session_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    Write-Host "train.txt is locked (close it in the editor) - logging to: $TrainLog" -ForegroundColor Yellow
    Set-Content -Path $TrainLog -Value $hdr -Encoding utf8
}
try {
    try { Stop-Transcript -ErrorAction SilentlyContinue } catch {}
    Start-Transcript -Path $TrainLog -Append | Out-Null
} catch {
    Write-Warning "Could not start transcript to $TrainLog - continuing without file log: $_"
}

try {
    # --- TEMP/TMP on D: (pip & installers need space; avoids C: full) ---
    $TmpDir = Join-Path $Root "temp_work"
    New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
    $env:TEMP = $TmpDir
    $env:TMP = $TmpDir

    # --- PyTorch / pip cache on D: ---
    $env:TORCH_HOME = Join-Path $Root ".torch"
    $env:XDG_CACHE_HOME = Join-Path $Root ".cache"
    New-Item -ItemType Directory -Force -Path $env:TORCH_HOME | Out-Null
    New-Item -ItemType Directory -Force -Path $env:XDG_CACHE_HOME | Out-Null

    # --- All pipeline outputs (splits, checkpoints, figures) on D: ---
    $env:PE_OUT_DIR = Join-Path $Root "pe_dl_outputs"

    # All segmentation epochs (no early exit): set PE_SEG_EARLY_STOP=0 before calling this script.

    # Dataset: prefer IEEE under Dataset And Files when present (91 volumes etc.)
    # Manual override:
    # $env:PE_IMG_DIR = "D:\my_project\Dataset And Files\IEEE dataset\images\images"
    # $env:PE_MASK_DIR = "D:\my_project\Dataset And Files\IEEE dataset\rs\rs"

    $ieeeImg = Join-Path $Root "Dataset And Files\IEEE dataset\images\images"
    $ieeeMsk = Join-Path $Root "Dataset And Files\IEEE dataset\rs\rs"
    $ieeeReady =
        (Test-Path $ieeeImg) -and (Test-Path $ieeeMsk) -and
        ((@(Get-ChildItem $ieeeImg -Filter "*.nrrd" -ErrorAction SilentlyContinue)).Count -gt 0)

    # Prefer IEEE when folder exists (also overrides stale env pointing at demo_nrrd)
    if ($ieeeReady -and ((-not $env:PE_IMG_DIR) -or ($env:PE_IMG_DIR -like "*demo_nrrd*"))) {
        $nrrdCount = @(Get-ChildItem $ieeeImg -Filter "*.nrrd" -ErrorAction SilentlyContinue).Count
        $env:PE_IMG_DIR = $ieeeImg
        $env:PE_MASK_DIR = $ieeeMsk
        Remove-Item Env:\PE_PATCH -ErrorAction SilentlyContinue
        Write-Host "Using IEEE dataset -> $ieeeImg ($nrrdCount .nrrd volumes)" -ForegroundColor Green
    }

    # Fallback: tiny synthetic demo only if IEEE not configured above
    if (-not $env:PE_IMG_DIR) {
        $demoImg = Join-Path $Root "pe_dl_research\demo_nrrd\images"
        if (Test-Path $demoImg) {
            $env:PE_IMG_DIR = $demoImg
            $env:PE_MASK_DIR = Join-Path $Root "pe_dl_research\demo_nrrd\masks"
            Write-Host "Using demo NRRD under pe_dl_research\demo_nrrd"
        }
    }
    # Synthetic demo volumes are 64^3; default patch 96^3 would fail RandCrop
    if ($env:PE_IMG_DIR -and ($env:PE_IMG_DIR -like "*demo_nrrd*")) {
        if (-not $env:PE_PATCH) {
            $env:PE_PATCH = "64,64,64"
            Write-Host "PE_PATCH -> 64,64,64 (demo volumes)"
        }
    }

    Write-Host "TEMP/TMP -> $TmpDir"
    Write-Host "PE_OUT_DIR -> $env:PE_OUT_DIR"
    if ($env:PE_IMG_DIR) { Write-Host "PE_IMG_DIR -> $env:PE_IMG_DIR" }
    if ($env:PE_MASK_DIR) { Write-Host "PE_MASK_DIR -> $env:PE_MASK_DIR" }
    Write-Host "TORCH_HOME -> $env:TORCH_HOME"
    Write-Host "VS Code / log file -> $TrainLog" -ForegroundColor Cyan

    Set-Location (Join-Path $Root "pe_dl_research")

    Write-Host "`n=== GPU check ==="
    python check_gpu.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nCUDA PyTorch missing - install GPU build first:" -ForegroundColor Yellow
        Write-Host "  .\install_cuda_pytorch.ps1"
        Write-Host "Or see INSTALL_GPU.md"
        exit 1
    }

    Write-Host "`n=== Preprocessing (splits) ==="
    python preprocessing.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n=== Train segmentation (GPU) ==="
    python train_segmentation.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n=== Train classification (GPU) ==="
    python train_classification.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`nDone. Outputs: $env:PE_OUT_DIR"
}
finally {
    $f = "=== Log ended $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    Add-Content -Path $TrainLog -Value $f -Encoding utf8 -ErrorAction SilentlyContinue
    Stop-TrainTranscript
}
