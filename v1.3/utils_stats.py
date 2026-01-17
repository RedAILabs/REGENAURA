# RED/regen/utils_stats.py
import torch
import torch.nn.functional as F
import math
from types import SimpleNamespace

EPS = 1e-8

def match_mean_std(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """
    Rescale `source` to have same mean/std as `target`.
    Non-destructive: returns new tensor.
    """
    s_mean, s_std = source.mean(), source.std(unbiased=False)
    t_mean, t_std = target.mean(), target.std(unbiased=False)

    s_std = s_std + EPS
    t_std = t_std + EPS

    scaled = (source - s_mean) / s_std * t_std + t_mean
    return scaled

def interpolate_2d_array(arr: torch.Tensor, new_shape: tuple):
    assert arr.dim() == 2, "interpolate_2d_array expects 2D tensors"
    device = arr.device
    dtype = arr.dtype
    x = arr.unsqueeze(0).unsqueeze(0).float().contiguous()  # [1,1,H,W]
    x_up = F.interpolate(x, size=new_shape, mode="bilinear", align_corners=False)
    return x_up.squeeze(0).squeeze(0).to(device=device, dtype=dtype)

def safe_noise(shape, device=None, scale=1e-3, dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=dtype) * scale

def count_parameters(state_dict: dict) -> int:
    """Count number of parameters from a state_dict (approx)."""
    total = 0
    for v in state_dict.values():
        if hasattr(v, "numel"):
            total += v.numel()
    return int(total)

def analyze_checkpoint(state_dict: dict):
    """
    Minimal analyzer for REGENAURA:
      - total_params : total param count
      - num_tensors  : number of tensor entries
      - largest_tensor_shape : (name, shape, numel)
    Returns a SimpleNamespace object with attributes used by regenaura.
    """
    total = count_parameters(state_dict)
    tensors = [(k, v.shape if hasattr(v, "shape") else None, getattr(v, "numel", lambda: 0)()) for k, v in state_dict.items() if hasattr(v, "numel")]
    num_tensors = len(tensors)
    largest = max(tensors, key=lambda x: x[2]) if tensors else (None, None, 0)
    return SimpleNamespace(total_params=int(total), num_tensors=num_tensors, largest_tensor=largest)
