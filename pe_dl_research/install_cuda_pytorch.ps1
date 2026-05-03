# Run in PowerShell:  .\install_cuda_pytorch.ps1
# Uses D:\pip_temp for pip extract if C: is low on space (edit path if needed).
$ErrorActionPreference = "Stop"
$pipTemp = "D:\my_project\temp_work"
New-Item -ItemType Directory -Force -Path $pipTemp | Out-Null
$env:TEMP = $pipTemp
$env:TMP = $pipTemp
Write-Host "TEMP/TMP -> $pipTemp"
python -m pip cache purge
python -m pip uninstall torch torchvision -y
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python check_gpu.py
