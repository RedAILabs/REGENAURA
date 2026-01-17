# RED/regen/aura_normalize.py
import torch
from utils_stats import match_mean_std

@torch.no_grad()
def aura_align_state_dicts(target_state: dict, source_state: dict, strict_keys: bool = False):
    """
    Align source_state parameter statistics onto target_state where shapes match.
    - target_state: reference (usually smaller/stable model)
    - source_state: to be adjusted (expanded model)
    Returns a modified copy of source_state (does NOT change input dict).
    """
    new_state = {}
    for k, v in source_state.items():
        if k in target_state and v.shape == target_state[k].shape:
            t = target_state[k].to(device=v.device, dtype=v.dtype)

            try:
                norm_t = match_mean_std(t, v)
                new_state[k] = norm_t.to(device=v.device, dtype=v.dtype)
            except Exception as e:
                print(f"[AURA-NORMALIZE] Failed to normalize {k}: {e}")
                new_state[k] = v.clone()
        else:
            # keep shape-mismatched tensors unchanged
            new_state[k] = v.clone()
    return new_state

@torch.no_grad()
def normalize_state_dict(state: dict):
    """
    Normalize every tensor in the state dict by subtracting mean
    and dividing by std, channel-wise.
    """
    new = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            new[k] = v
            continue

        mean = v.mean()
        std = v.std(unbiased=False) + 1e-8
        new[k] = (v - mean) / std
    return new


def analyze_checkpoint(state: dict):
    """Minimal stub so regenaura stats are not broken."""
    class Dummy:
        total_params = sum(v.numel() for v in state.values())
    return Dummy()
