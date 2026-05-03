"""
GPU selection and status for training (8GB+ NVIDIA).
- Prefers CUDA; honors PE_FORCE_CPU=1 to force CPU.
- PE_REQUIRE_GPU=1 raises if CUDA is missing (avoids slow CPU by mistake).
"""
from __future__ import annotations

import os
from typing import Any

import torch


def print_device_banner() -> torch.device:
    """Resolve device, print one-line status, return torch.device."""
    if os.environ.get("PE_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        print("[device] Forced CPU (PE_FORCE_CPU=1)")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print("=" * 60)
        print("[device] WARNING: CUDA not found — training will use CPU (very slow).")
        print("  For an NVIDIA GPU, install the CUDA build of PyTorch, e.g.:")
        print("  pip uninstall torch torchvision -y")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("  (Use cu121 / cu118 from pytorch.org if your driver needs it.)")
        print("  Then: python -c \"import torch; print(torch.cuda.is_available())\"  # should be True")
        print("=" * 60)
        if os.environ.get("PE_REQUIRE_GPU", "").lower() in ("1", "true", "yes"):
            raise RuntimeError("PE_REQUIRE_GPU=1 but torch.cuda.is_available() is False.")
        return torch.device("cpu")

    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        try:
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        except Exception:
            total_gb = 0.0
        cap = torch.cuda.get_device_capability(i)
        print(f"[device] GPU {i}: {name}  ~{total_gb:.1f} GiB  (CC {cap[0]}.{cap[1]})")
    try:
        ver = torch.version.cuda
        print(f"[device] PyTorch built with CUDA: {ver}")
    except Exception:
        pass
    return torch.device("cuda:0")


def make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """GradScaler for AMP on CUDA; disabled scaler on CPU (safe no-op path)."""
    use_amp = enabled and device.type == "cuda"
    if device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=use_amp)
            except Exception:
                from torch.cuda.amp import GradScaler

                return GradScaler(enabled=use_amp)
    from torch.cuda.amp import GradScaler

    return GradScaler(enabled=False)


def autocast_if_cuda(device: torch.device):
    """Mixed precision on CUDA only."""
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        try:
            from torch.cuda.amp import autocast as cuda_autocast

            return cuda_autocast()
        except Exception:
            from contextlib import nullcontext

            return nullcontext()
