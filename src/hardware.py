import os
import platform
import subprocess

def detect_hardware():
    info = {
        "platform": platform.platform(),
        "ram_gb": None,
        "gpu_name": None,
        "vram_gb": 0.0,
        "cuda_available": False,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass

    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
    except Exception:
        pass
    return info

def choose_model(info):
    # Keep selection conservative: FLUX is intentionally not auto-selected because
    # it has much higher memory/storage needs and model access/license requirements vary.
    vram = float(info.get("vram_gb") or 0)
    ram = float(info.get("ram_gb") or 0)
    if info.get("cuda_available") and vram >= 12:
        return {
            "name": "Stable Diffusion XL Base 1.0",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "width": 1024, "height": 1024, "steps": 25,
            "dtype": "float16", "use_sdxl": True,
        }
    if info.get("cuda_available") and vram >= 6:
        return {
            "name": "Stable Diffusion 1.5",
            "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "width": 512, "height": 512, "steps": 25,
            "dtype": "float16", "use_sdxl": False,
        }
    # CPU or very small GPU: use SD 1.5 at low resolution; generation may be slow.
    return {
        "name": "Stable Diffusion 1.5 (low-memory fallback)",
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "width": 384, "height": 384, "steps": 15,
        "dtype": "float32", "use_sdxl": False,
    }
