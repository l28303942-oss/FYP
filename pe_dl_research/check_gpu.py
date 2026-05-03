"""Quick check: PyTorch sees your NVIDIA GPU. Run: python check_gpu.py"""
import torch

print("torch.__version__", torch.__version__)
print("cuda.is_available()", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    m = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM (reported) ~ {m:.1f} GiB")
else:
    print("No CUDA — install GPU build, see INSTALL_GPU.md")
