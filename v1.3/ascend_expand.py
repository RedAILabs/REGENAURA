# RED/regen/ascend_expand.py
"""
REGENAURA ASCEND: prototype expansion utilities.

Key functions:
 - detect_target_from_checkpoint: infer growth target from args
 - expand_embedding_matrix
 - expand_linear_weight
 - expand_attention_heads (qkv splitting/tiling strategy)
 - upscale_state_dict: orchestrator that returns a new state_dict sized for `target_param_count` (approx)
"""

import math
import copy
import torch
from utils_stats import count_parameters, interpolate_2d_array, safe_noise

def detect_param_count(state_dict):
    return count_parameters(state_dict)

def compute_scale_factor(src_count: int, tgt_count: int):
    """Return float scale factor (tgt/src)."""
    return float(tgt_count) / max(1, float(src_count))

# ---------------- utilities ----------------

def expand_embedding_matrix(old_emb: torch.Tensor, target_vocab_size: int, method="interpolate", device=None):
    """
    Expand token embedding matrix from (vocab_old, dim) -> (vocab_new, dim).
    Methods: 'tile', 'interpolate', 'pad'
    """
    old_emb = old_emb.to(device=device)
    v_old, d = old_emb.shape
    v_new = int(target_vocab_size)
    if v_new == v_old:
        return old_emb.clone()

    if method == "tile":
        repeats = math.ceil(v_new / v_old)
        out = old_emb.repeat(repeats, 1)[:v_new, :].clone()
        return out

    if method == "interpolate":
        # interpolate along vocab dimension
        out = interpolate_2d_array(old_emb, (v_new, d))
        return out.to(device=old_emb.device)

    # default: pad with small noise
    out = torch.zeros((v_new, d), device=old_emb.device, dtype=old_emb.dtype)
    out[:v_old] = old_emb
    out[v_old:] = safe_noise((v_new - v_old, d), device=old_emb.device, scale=1e-3, dtype=old_emb.dtype)
    return out

def expand_linear_weight(W_old: torch.Tensor, new_out: int, new_in: int, method="tile", noise_scale=1e-3):
    """
    Expand a 2D linear weight [out, in] -> [new_out, new_in].
    Strategies:
      - tile: repeat existing rows/cols
      - interpolate: use bilinear interpolation via utils.interpolate_2d_array
      - pad: copy then pad
    """
    W_old = W_old.detach()
    old_out, old_in = W_old.shape
    if new_out == old_out and new_in == old_in:
        return W_old.clone()

    # preferred: interpolate
    try:
        new = interpolate_2d_array(W_old, (new_out, new_in))
        return new
    except Exception:
        pass

    # fallback: tile rows/cols then trim
    out = W_old
    if new_out != old_out:
        repeats = math.ceil(new_out / old_out)
        out = out.repeat(repeats, 1)[:new_out, :]
    if new_in != old_in:
        repeats = math.ceil(new_in / old_in)
        out = out.repeat(1, repeats)[:, :new_in]
    # add tiny noise to new regions
    if new_out > old_out or new_in > old_in:
        mask = torch.ones_like(out)
        mask[:old_out, :old_in] = 0.0
        out = out + safe_noise(out.shape, device=out.device, scale=noise_scale, dtype=out.dtype) * mask
    return out

def expand_attention_qkv(param_qkv: torch.Tensor, old_num_heads: int, new_num_heads: int, head_dim: int):
    """
    param_qkv often has shape [embed_dim, 3*embed_dim] or concatenated QKV differently in some models.
    Here we expect param shape of [3*H*d, H*d] or [H*d, 3*H*d] variants differ.
    This function provides a simple strategy to clone and tile heads to reach new head count.
    NOTE: This is heuristic and must be adapted to your model's exact q/k/v layout.
    """
    # If qkv is 2D and shaped [out, in], treat as linear and expand simply
    if param_qkv.dim() == 2:
        out, inp = param_qkv.shape
        # attempt interpolation
        try:
            return expand_linear_weight(param_qkv, out, inp)
        except Exception:
            return param_qkv.clone()

    return param_qkv.clone()

# ---------------- orchestrator ----------------

def upscale_state_dict(
    src_state: dict,
    prototype_target_params: int = None,
    explicit_target_state: dict = None,
    prefer_vocab_method="interpolate",
    device="cpu"
):
    """
    Generate a new state dict expanded to approximately `prototype_target_params`.
    If `explicit_target_state` is provided (e.g., a real larger checkpoint),
    use it as shape template (preferred).
    Returns: new_state_dict (torch tensors)
    """
    src = copy.deepcopy(src_state)
    new_state = {}

    src_count = detect_param_count(src)
    print(f"[REGEN] Source param count ~ {src_count:,}")

    if explicit_target_state is not None:
        tgt_count = detect_param_count(explicit_target_state)
        print(f"[REGEN] Using explicit target checkpoint -> param count ~ {tgt_count:,}")
        scale = compute_scale_factor(src_count, tgt_count)
    elif prototype_target_params is not None:
        tgt_count = int(prototype_target_params)
        scale = compute_scale_factor(src_count, tgt_count)
        print(f"[REGEN] Target param count set -> {tgt_count:,} scale {scale:.3f}")
    else:
        raise ValueError("Provide prototype_target_params or explicit_target_state.")

    # If scale is ~1.0, nothing to do
    if abs(scale - 1.0) < 0.05:
        print("[REGEN] Scale close to 1.0 — copying state dict")
        return {k: v.clone() for k, v in src.items()}

    # Heuristic: map shapes using explicit target if available, else grow dims by scale factor
    for k, v in src.items():
        v_cpu = v.to(device)
        if explicit_target_state and k in explicit_target_state:
            tgt_tensor = explicit_target_state[k].to(device)
            # If shapes match, copy/align via tensor interpolation if needed
            if tgt_tensor.shape == v_cpu.shape:
                new_state[k] = v_cpu.clone()
            else:
                # Try expand logically
                if v_cpu.dim() == 2 and tgt_tensor.dim() == 2:
                    new_state[k] = expand_linear_weight(v_cpu, tgt_tensor.shape[0], tgt_tensor.shape[1])
                elif v_cpu.dim() == 1 and tgt_tensor.dim() == 1:
                    # bias/1d param: tile or pad
                    new = torch.zeros_like(tgt_tensor)
                    new[:v_cpu.shape[0]] = v_cpu
                    if tgt_tensor.shape[0] > v_cpu.shape[0]:
                        new[v_cpu.shape[0]:] = safe_noise((tgt_tensor.shape[0]-v_cpu.shape[0],), device=device, scale=1e-4)
                    new_state[k] = new
                else:
                    # fallback: try to reshape safely
                    try:
                        new_state[k] = expand_linear_weight(v_cpu.view(-1, 1), tgt_tensor.numel(), 1).view(tgt_tensor.shape)
                    except Exception:
                        new_state[k] = tgt_tensor.clone()
        else:
            # No explicit target shape -> expand by scale heuristics
            if v_cpu.dim() == 2:
                new_out = max(1, int(round(v_cpu.shape[0] * math.sqrt(scale))))
                new_in = max(1, int(round(v_cpu.shape[1] * math.sqrt(scale))))
                new_state[k] = expand_linear_weight(v_cpu, new_out, new_in)
            elif v_cpu.dim() == 1:
                new_len = max(1, int(round(v_cpu.shape[0] * (scale ** 0.5))))
                new = torch.zeros(new_len, device=device, dtype=v_cpu.dtype)
                new[:v_cpu.shape[0]] = v_cpu
                if new_len > v_cpu.shape[0]:
                    new[v_cpu.shape[0]:] = safe_noise((new_len - v_cpu.shape[0],), device=device, scale=1e-4, dtype=v_cpu.dtype)
                new_state[k] = new
                # safe fallback for dims > 2
            else:
                try:
                    # fold trailing dims into 2D and try expand
                    flat = v_cpu.view(v_cpu.shape[0], -1)  # [dim0, rest]
                    new_flat = expand_linear_weight(flat, max(1, int(round(flat.shape[0] * scale))), flat.shape[1])
                    # reshape or pad/truncate safely
                    if new_flat.numel() >= v_cpu.numel():
                        new_arr = new_flat.flatten()[:v_cpu.numel()].view(v_cpu.shape)
                        new_state[k] = new_arr
                    else:
                        tmp = torch.zeros_like(v_cpu)
                        tmp.view(-1)[:new_flat.numel()] = new_flat.flatten()
                        new_state[k] = tmp
                except Exception:
                    new_state[k] = v_cpu.clone()

    new_count = detect_param_count(new_state)
    print(f"[REGEN] New param count ~ {new_count:,}")
    return new_state

# ---------------- helper example ----------------

def run_upscale_example(src_ckpt_path, out_path, prototype_target=700_000_000, explicit_target_ckpt=None, device="cpu"):
    """
    Example usage:
       run_upscale_example("450m.pt", "450m_to_700m.pt")
    """
    import torch
    src = torch.load(src_ckpt_path, map_location="cpu")
    explicit_target = None
    if explicit_target_ckpt:
        explicit_target = torch.load(explicit_target_ckpt, map_location="cpu")
    new_state = upscale_state_dict(src, prototype_target_params=prototype_target, explicit_target_state=explicit_target, device=device)
    torch.save(new_state, out_path)
    print("[REGEN] Saved expanded checkpoint:", out_path)

def ascend_expand(
        teacher_state: dict,
        target_params: int = None,
        explicit_target_state: dict = None,
        device: str = "cpu"
):
    """
    REGENAURA's official upscale wrapper.

    REGENAURA always calls:
        ascend_expand(teacher_state, target_params...)

    We simply redirect that call to upscale_state_dict(),
    which is the actual upscaling engine in this file.
    """

    if not isinstance(teacher_state, dict):
        raise ValueError("ascend_expand: teacher_state must be a state_dict")

    # Case 1: Use explicit target checkpoint
    if explicit_target_state is not None:
        return upscale_state_dict(
            teacher_state,
            prototype_target_params=None,
            explicit_target_state=explicit_target_state,
            device=device
        )

    # Case 2: Use target parameter count
    if target_params is None:
        raise ValueError("ascend_expand requires either target_params or explicit_target_state.")

    return upscale_state_dict(
        teacher_state,
        prototype_target_params=target_params,
        explicit_target_state=None,
        device=device
    )
