# File: regenaura.py  (PATCHED)
"""
REGENAURA v1.1 – patched orchestrator

This version:
 - adapts imports to the real module APIs in your repo
 - provides unified wrappers for AURASCEND / DESCEND so the rest of the file can use stable calls
 - removes stale AuraStats type hints
 - safer teacher_model reconstruction for distillation
"""

from __future__ import annotations

import os
import json
import copy
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, List, Tuple, Dict, Any, Iterable, Sequence
from types import SimpleNamespace
import glob
import os
import gc

import torch
from torch import nn

from aura_normalize import aura_align_state_dicts, analyze_checkpoint, normalize_state_dict
from descend_distill import descend_soft_distill
from ascend_manual import ManualScaler
from ascend_hybrid import AscendHybrid
from ascend_auto import AutoScaler
from ascend_expand import ascend_expand
from utils_stats import count_parameters

def ascend_manual(base_config):
    return ManualScaler(base_config)

def ascend_hybrid(base_config):
    return AscendHybrid(base_config)

def ascend_auto(engine):
    return AutoScaler(engine)

def summarize_model_stats(state_dict: Dict[str, Any]):
    total = int(count_parameters(state_dict))
    return SimpleNamespace(total_params=total)

# PyTorch 2.6 pickle-safe allowlist for GradScaler (needed for some .pt files)
from torch.serialization import add_safe_globals
from torch.cuda.amp.grad_scaler import GradScaler
add_safe_globals([GradScaler])

# ------------------------------------------
# dataclasses & config (unchanged)
# ------------------------------------------

RegenMode = Literal[
    "ascend_manual",
    "ascend_auto",
    "ascend_expand",
    "ascend_hybrid",
    "distill_only",
]

PrecisionMode = Literal["fp32", "fp16", "bf16"]


@dataclass
@dataclass
class RegenDistillConfig:
    steps: int = 500
    temperature: float = 1.0
    alpha_kl: float = 0.7
    alpha_ce: float = 0.3
    max_grad_norm: float = 1.0
    use_amp: bool = True
    precision: PrecisionMode = "bf16"
    log_every: int = 10

@dataclass
class RegenAscendConfig:
    mode: RegenMode = "ascend_auto"
    target_hidden_size: Optional[int] = None
    target_num_layers: Optional[int] = None
    width_gain: Optional[float] = None
    depth_gain: Optional[float] = None
    max_params: Optional[int] = None
    max_vram_gb: Optional[float] = None
    prefer_depth_over_width: bool = False


@dataclass
class RegenConfig:
    ascend: Optional[RegenAscendConfig] = None
    distill: Optional[RegenDistillConfig] = None
    output_dir: str = "./regen_output"
    run_name: str = "regen_run"
    normalize_teacher: bool = True
    normalize_student_init: bool = False
    save_json_summary: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # optional: user can attach a dataloader here for distillation
    dataloader: Optional[Any] = None


@dataclass
class RegenRunSummary:
    teacher_ckpt: str
    student_ckpt: str
    mode: RegenMode
    device: str
    teacher_param_count: int
    student_param_count: int
    ascend_used: bool
    distill_used: bool
    extra: Dict[str, Any]


# ------------------------------------------
# internal helpers copied/adapted from original
# ------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# -----------------------------
# PYTORCH 2.6 SAFE GLOBAL FIX
# -----------------------------
# Needed because your .pt checkpoint includes AMP GradScaler

# ==========================================================
# 1) SAFE SEQUENTIAL CHECKPOINT APPLIER (NO MERGING)
# ==========================================================
def apply_checkpoints_in_place(
    model: torch.nn.Module,
    ckpt_paths: Sequence[str],
    device: str = "cpu",
    overwrite: bool = False,
    verbose: bool = True,
):
    """
    Apply checkpoints sequentially WITHOUT merging.
    Streams safetensors one-key-at-a-time if safetensors.safe_open is available.
    """
    model_sd = model.state_dict()
    applied = 0
    skipped = 0
    unknown = 0

    for p in ckpt_paths:
        if verbose:
            print(f"[apply_ckpt] Loading checkpoint: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        # If safetensors, try safe_open (stream)
        if p.endswith(".safetensors"):
            try:
                from safetensors import safe_open as st_safe_open
                with st_safe_open(p, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k not in model_sd:
                            unknown += 1
                            if verbose and unknown <= 20:
                                print(f"[apply_ckpt] key '{k}' not found in model → skipping")
                            continue
                        if not overwrite:
                            skipped += 1
                            continue
                        t = f.get_tensor(k)  # this loads only this tensor
                        t = t.to(device=device)
                        try:
                            model_sd[k].copy_(t.to(model_sd[k].device, dtype=model_sd[k].dtype))
                            applied += 1
                        except Exception as e:
                            print(f"[apply_ckpt] Failed to copy key {k}: {e}")
                            raise
                # free caches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if verbose:
                    print(f"[apply_ckpt] Finished {p}: applied={applied}, skipped={skipped}, unknown={unknown}")
                continue
            except Exception as e:
                # fallback to load_file for older safetensors versions
                if verbose:
                    print(f"[apply_ckpt] safe_open streaming failed for {p}: {e}; falling back to load_file")

        # Fallback: load entire file into memory (torch or safetensors.load_file)
        try:
            if p.endswith(".safetensors"):
                from safetensors.torch import load_file as _safetensors_load
                loaded = _safetensors_load(p)
            else:
                loaded = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {p}: {e}") from e

        # unwrap any wrappers
        if isinstance(loaded, dict) and ("model_state_dict" in loaded or "state_dict" in loaded):
            loaded_state = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
        else:
            loaded_state = loaded

        if not isinstance(loaded_state, dict):
            if verbose:
                print(f"[apply_ckpt] File {p} did not contain a state-dict; skipping.")
            continue

        for k, v in loaded_state.items():
            if k not in model_sd:
                unknown += 1
                if verbose and unknown <= 20:
                    print(f"[apply_ckpt] key '{k}' not found in model → skipping")
                continue
            if not overwrite:
                skipped += 1
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            try:
                v = v.to(device=device)
                model_sd[k].copy_(v.to(model_sd[k].device, dtype=model_sd[k].dtype))
                applied += 1
            except Exception as e:
                print(f"[apply_ckpt] Failed to copy key {k}: {e}")
                raise
        del loaded_state
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"[apply_ckpt] Finished {p}: applied={applied}, skipped={skipped}, unknown={unknown}")

    try:
        model.load_state_dict(model_sd, strict=False)
    except Exception:
        pass

    return {"applied": applied, "skipped": skipped, "unknown": unknown}

# ==========================================================
# 2) SINGLE-FILE CHECKPOINT LOADER (NO MERGING!)
# ==========================================================
def _load_checkpoint(path: Union[str, List[str]], map_location="cpu") -> Dict[str, Any]:
    """
    Simplified loader:
      - Accepts only ONE checkpoint file path (string)
      - Does NOT merge shards
      - Used ONLY for Red-70M or single-file teacher checkpoints

    DeepSeek or other multi-shard models must use apply_checkpoints_in_place().
    """

    # if user accidentally provides list → error
    if isinstance(path, (list, tuple)):
        if len(path) == 1:
            path = path[0]
        else:
            raise ValueError("Multiple checkpoint files detected. Use apply_checkpoints_in_place() instead.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # ---------- Load safetensors ----------
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            loaded = load_file(path)
        except Exception:
            loaded = torch.load(path, map_location=map_location, weights_only=False)
    else:
        # ---------- Load .pt / .pth / .bin ----------
        loaded = torch.load(path, map_location=map_location, weights_only=False)

    # unwrap inner dict
    if isinstance(loaded, dict) and ("model_state_dict" in loaded or "state_dict" in loaded):
        return loaded.get("model_state_dict", loaded.get("state_dict", loaded))

    if isinstance(loaded, dict):
        return loaded

    raise RuntimeError(f"Unsupported checkpoint structure in {path}")

def _get_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _maybe_analyze_checkpoint(state_dict: Dict[str, Any]):
    """
    Return the result of analyze_checkpoint(state_dict) if available,
    otherwise return a minimal stats object with total_params.
    """
    if analyze_checkpoint is None:
        print("[REGENAURA] aura_normalize.analyze_checkpoint not available – skipping stats.")
        total = sum(getattr(v, "numel", lambda: 0)() for v in state_dict.values() if hasattr(v, "numel"))
        return SimpleNamespace(total_params=int(total))
    try:
        return analyze_checkpoint(state_dict)
    except Exception as e:
        print(f"[REGENAURA] analyze_checkpoint failed: {e}")
        total = sum(getattr(v, "numel", lambda: 0)() for v in state_dict.values() if hasattr(v, "numel"))
        return SimpleNamespace(total_params=int(total))


def _maybe_normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if normalize_state_dict is None:
        return state_dict
    try:
        return normalize_state_dict(state_dict)
    except Exception as e:
        print(f"[REGENAURA] normalize_state_dict failed: {e}")
        return state_dict


def _apply_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        # some state-dicts might be nested; try to handle common wrapper keys
        try:
            if "model_state_dict" in state_dict:
                missing, unexpected = model.load_state_dict(state_dict["model_state_dict"], strict=False)
            else:
                raise
        except Exception as e2:
            print(f"[REGENAURA] Failed loading state_dict: {e2}")
            return

    if missing:
        print(f"[REGENAURA] Missing keys when loading student init: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[REGENAURA] Unexpected keys when loading student init: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")


def _select_ascend_fn(mode: RegenMode):
    if mode == "ascend_manual":
        return ascend_manual
    if mode == "ascend_auto":
        return ascend_auto
    if mode == "ascend_expand":
        return ascend_expand
    if mode == "ascend_hybrid":
        return ascend_hybrid
    return None


# ------------------------------------------
# Unified wrappers for ascend/distill
# ------------------------------------------

def _run_ascend(
    mode: RegenMode,
    teacher_stats,
    student_model: nn.Module,
    ascend_cfg: RegenAscendConfig,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Normalize calls to various ascend_* implementations to a single API.
    Returns (student_model, ascend_info).
    """
    if mode == "ascend_manual":
        if ascend_manual is None:
            raise RuntimeError("ascend_manual not available")
        base_config = {
            "hidden_size": getattr(getattr(student_model, "config", SimpleNamespace()), "d_model", getattr(student_model, "config", {}).get("hidden_size", None)),
            "num_layers": getattr(getattr(student_model, "config", SimpleNamespace()), "num_layers", None),
        }
        scaler = ascend_manual(base_config)
        new_arch = scaler.set(ascend_cfg.width_gain or 0.0, ascend_cfg.depth_gain or 0.0)
        return student_model, {"target_arch": new_arch}

    if mode == "ascend_auto":
        if ascend_auto is None or ascend_hybrid is None:
            raise RuntimeError("ascend_auto or ascend_hybrid missing")
        base_config = {
            "hidden_size": getattr(getattr(student_model, "config", SimpleNamespace()), "d_model", getattr(student_model, "config", {}).get("hidden_size", None)),
            "num_layers": getattr(getattr(student_model, "config", SimpleNamespace()), "num_layers", None),
        }
        engine = ascend_hybrid(base_config)
        auto_engine = ascend_auto(engine)
        # auto_engine.adjust() may return (new_arch, score) or similar
        try:
            new_arch, score = auto_engine.adjust()
        except Exception:
            new_arch = engine.compute_new_architecture()
            score = 0.0
        return student_model, {"target_arch": new_arch, "auto_score": score}

    if mode == "ascend_expand":
        if ascend_expand is None:
            raise RuntimeError("ascend_expand missing")
        if ascend_cfg.max_params is None:
            raise ValueError("ascend_expand requires ascend_cfg.max_params")
        # use student_model.state_dict() as source
        src_state = student_model.state_dict()

        new_state = ascend_expand(
            src_state,
            target_params=ascend_cfg.max_params,
            explicit_target_state=None,
            device=device
        )

        # --- SANITY CHECK: ENSURE NO NAN/INF ---
        for k, v in new_state.items():
            if not torch.isfinite(v).all():
                print(f"[REGENAURA][WARN] Non-finite tensor in {k}, zeroing")
                new_state[k] = torch.zeros_like(v)

        # --- NEW PATCH: rebuild model layers before loading ---
        try:
            from rebuild_student_linears_from_state import rebuild_student_linears_from_state
            student_model = rebuild_student_linears_from_state(student_model, new_state)
        except Exception as e:
            print("[REGENAURA] WARNING: model rebuild failed:", e)

        # Now load the state
        try:
            student_model.load_state_dict(new_state, strict=False)
        except Exception as e:
            print(f"[REGENAURA] Warning: loading expanded state failed even after rebuild: {e}")

        return student_model, {"target_params": ascend_cfg.max_params}

    if mode == "ascend_hybrid":
        if ascend_hybrid is None:
            raise RuntimeError("ascend_hybrid missing")
        base_config = {
            "hidden_size": getattr(getattr(student_model, "config", SimpleNamespace()), "d_model", getattr(student_model, "config", {}).get("hidden_size", None)),
            "num_layers": getattr(getattr(student_model, "config", SimpleNamespace()), "num_layers", None),
        }
        engine = ascend_hybrid(base_config)
        engine.set_scaling(ascend_cfg.width_gain or 0.0, ascend_cfg.depth_gain or 0.0)
        new_arch = engine.compute_new_architecture()
        return student_model, {"target_arch": new_arch}

    raise RuntimeError(f"Unknown ascend mode '{mode}'")


def _run_distill(
    teacher_state_dict: Dict[str, Any],
    student_model: nn.Module,
    regen_cfg: RegenConfig,
    tokenizer,
    device: str,
) -> nn.Module:
    """
    Normalize distillation call to descend_soft_distill.
    This function attempts to build a teacher_model from the teacher_state_dict
    (by cloning the student class or loading into a copy), finds a dataloader,
    and calls descend_soft_distill.
    """
    if descend_soft_distill is None:
        raise RuntimeError("descend_soft_distill not available")

    # find dataloader: prefer regen_cfg.dataloader, then regen_cfg.__dict__['dataloader'], otherwise error
    dataloader = getattr(regen_cfg, "dataloader", None)
    if dataloader is None:
        # maybe user placed it in regen_cfg.extra (legacy)
        extra = getattr(regen_cfg, "extra", None)
        if isinstance(extra, dict):
            dataloader = extra.get("dataloader", None)

    if dataloader is None:
        raise RuntimeError("Distillation requires a dataloader. Attach it to regen_cfg.dataloader")

    # Attempt to create a teacher model instance:
    teacher_model = None
    try:
        # try to instantiate same class if possible: many models accept config as constructor arg
        cls = student_model.__class__
        cfg = getattr(student_model, "config", None)
        if cfg is not None:
            try:
                teacher_model = cls(cfg).to(device)
            except Exception:
                # fallback: deepcopy student and load state
                teacher_model = copy.deepcopy(student_model).to(device)
        else:
            teacher_model = copy.deepcopy(student_model).to(device)
    except Exception:
        teacher_model = copy.deepcopy(student_model).to(device)

    # Load teacher weights into teacher_model
    try:
        # teacher_state_dict might be wrapped
        if "model_state_dict" in teacher_state_dict:
            teacher_weights = teacher_state_dict["model_state_dict"]
        else:
            teacher_weights = teacher_state_dict
        teacher_model.load_state_dict(teacher_weights, strict=False)
    except Exception as e:
        print(f"[REGENAURA] Warning: teacher state load failed: {e} (continuing with best-effort teacher)")

    # Use distill config
    if regen_cfg.distill is None:
        raise RuntimeError("regen_cfg.distill is None but distillation requested")

    dist = regen_cfg.distill

    # call descend_soft_distill
    student_after = descend_soft_distill(
        teacher_model,
        student_model,
        dataloader,
        device=device,
        alpha_kl=dist.alpha_kl,
        beta_ce=dist.alpha_ce,
        temperature=dist.temperature,
        steps=dist.steps,
    )

    return student_after


# ------------------------------------------
# Public API – main orchestration entrypoints
# ------------------------------------------

def regenaura_from_checkpoint(
    teacher_ckpt_path: str,
    student_model: nn.Module,
    regen_cfg: RegenConfig,
    *,
    student_init_ckpt: Optional[str] = None,
    tokenizer: Optional[Any] = None,
) -> RegenRunSummary:
    device = regen_cfg.device
    student_model.to(device)

    # --- NEW PATCH: resume previous student checkpoint if it exists ---
    prev_ckpt = os.path.join(regen_cfg.output_dir, f"{regen_cfg.run_name}_student.pt")

    if os.path.isfile(prev_ckpt):
        print(f"[REGENAURA] Resuming from previous student checkpoint: {prev_ckpt}")
        try:
            prev = torch.load(prev_ckpt, map_location=device, weights_only=False)
            if isinstance(prev, dict) and "model_state_dict" in prev:
                student_model.load_state_dict(prev["model_state_dict"], strict=False)
            else:
                student_model.load_state_dict(prev, strict=False)
        except Exception as e:
            print(f"[REGENAURA] WARNING: Failed to load previous student checkpoint: {e}")
    else:
        print("[REGENAURA] No previous student checkpoint found – starting fresh.")

    print(f"[REGENAURA] Loading teacher checkpoint from: {teacher_ckpt_path}")
    teacher_ckpt = _load_checkpoint(teacher_ckpt_path, map_location=device)

    teacher_state = teacher_ckpt.get("model_state_dict", teacher_ckpt)

    teacher_stats = _maybe_analyze_checkpoint(teacher_state)
    if regen_cfg.normalize_teacher:
        print("[REGENAURA] Normalizing teacher checkpoint (AURA pass)...")
        teacher_state = _maybe_normalize_state_dict(teacher_state)

    teacher_param_count = None
    if summarize_model_stats is not None:
        try:
            teacher_param_count = teacher_stats.total_params  # type: ignore[attr-defined]
        except Exception:
            teacher_param_count = None

    ascend_used = False
    ascend_info = {}
    if regen_cfg.ascend and regen_cfg.ascend.mode != "distill_only":
        ascend_used = True
        ascend_cfg = regen_cfg.ascend
        print(f"[REGENAURA] Running AURASCEND ({ascend_cfg.mode})...")
        student_model, ascend_info = _run_ascend(
            ascend_cfg.mode,
            teacher_stats,
            student_model,
            ascend_cfg,
            device,
        )
        print(f"[REGENAURA] AURASCEND complete. Info: {ascend_info}")

    # Initialize student weights (optional)
    if student_init_ckpt is not None and os.path.isfile(student_init_ckpt):
        print(f"[REGENAURA] Loading student init checkpoint: {student_init_ckpt}")
        stu_ckpt = _load_checkpoint(student_init_ckpt, map_location=device)
        stu_state = stu_ckpt.get("model_state_dict", stu_ckpt)
        if regen_cfg.normalize_student_init:
            print("[REGENAURA] Normalizing student init weights...")
            stu_state = _maybe_normalize_state_dict(stu_state)
        _apply_state_dict(student_model, stu_state)
    else:
        print("[REGENAURA] No student init checkpoint provided – using current weights.")

    student_param_count = _get_param_count(student_model)

    distill_used = False
    distill_info = {}

    if regen_cfg.distill is not None and descend_soft_distill is not None:
        distill_used = True
        print("[REGENAURA] Starting DESCEND soft distillation...")
        student_model = _run_distill(
            teacher_state_dict=teacher_state,
            student_model=student_model,
            regen_cfg=regen_cfg,
            tokenizer=tokenizer,
            device=device,
        )
        distill_info = {
            "steps": regen_cfg.distill.steps,
            "temperature": regen_cfg.distill.temperature,
            "alpha_kl": regen_cfg.distill.alpha_kl,
            "alpha_ce": regen_cfg.distill.alpha_ce,
            "precision": regen_cfg.distill.precision,
        }
    else:
        print("[REGENAURA] Distillation disabled or descend_soft_distill unavailable.")

    # Save final student checkpoint + JSON summary
    _ensure_dir(regen_cfg.output_dir)
    student_ckpt_path = os.path.join(regen_cfg.output_dir, f"{regen_cfg.run_name}_student.pt")

    save_payload = {
        "model_state_dict": student_model.state_dict(),
        "regen_config": asdict(regen_cfg),
        "ascend_info": ascend_info,
        "distill_info": distill_info,
    }

    torch.save(save_payload, student_ckpt_path)
    print(f"[REGENAURA] Saved student checkpoint to: {student_ckpt_path}")

    extra = {"ascend_info": ascend_info, "distill_info": distill_info, "teacher_stats_available": teacher_stats is not None}

    summary = RegenRunSummary(
        teacher_ckpt=teacher_ckpt_path,
        student_ckpt=student_ckpt_path,
        mode=(regen_cfg.ascend.mode if regen_cfg.ascend else "distill_only"),  # type: ignore[arg-type]
        device=device,
        teacher_param_count=teacher_param_count or -1,
        student_param_count=student_param_count,
        ascend_used=ascend_used,
        distill_used=distill_used,
        extra=extra,
    )

    if regen_cfg.save_json_summary:
        json_path = os.path.join(regen_cfg.output_dir, f"{regen_cfg.run_name}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"[REGENAURA] Saved run summary to: {json_path}")

    return summary

def rebuild_student_linears_from_state(student_model: nn.Module, new_state: Dict[str, torch.Tensor]):
    """
    Rebuilds mismatched layers safely.
    Supports: Linear, LayerNorm, Embedding, Conv1D (GPT), and fallback tensor bypass.
    """

    replaced = []
    skipped = []

    for key, tensor in new_state.items():

        # Only operate on common learnable params
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue

        shape = tensor.shape
        mod_path = key.rsplit(".", 1)[0]
        parts = mod_path.split(".")

        obj = student_model
        parent = None
        attr = None

        # Walk the module tree
        try:
            for p in parts:
                parent = obj
                if p.isdigit():
                    idx = int(p)
                    obj = obj[idx]
                    attr = idx
                else:
                    obj = getattr(obj, p)
                    attr = p
        except Exception:
            skipped.append(key)
            continue

        old_mod = obj

        # Case 1: Linear
        if isinstance(old_mod, nn.Linear) and tensor.dim() == 2:
            out_f, in_f = tensor.shape
            print(f"[rebuild] Linear reset {mod_path}: in={in_f}, out={out_f}")

            new_mod = nn.Linear(in_f, out_f, bias=(old_mod.bias is not None))

            if isinstance(attr, int):
                parent[attr] = new_mod
            else:
                setattr(parent, attr, new_mod)

            replaced.append(mod_path)
            continue

        # Case 2: Embedding
        if isinstance(old_mod, nn.Embedding) and tensor.dim() == 2:
            num, dim = tensor.shape
            print(f"[rebuild] Embedding reset {mod_path}: num={num}, dim={dim}")
            new_mod = nn.Embedding(num, dim)

            if isinstance(attr, int):
                parent[attr] = new_mod
            else:
                setattr(parent, attr, new_mod)

            replaced.append(mod_path)
            continue

        # Case 3: LayerNorm
        if isinstance(old_mod, nn.LayerNorm) and tensor.dim() == 1:
            print(f"[rebuild] LayerNorm reset {mod_path}: dim={tensor.numel()}")
            new_mod = nn.LayerNorm(tensor.numel())

            if isinstance(attr, int):
                parent[attr] = new_mod
            else:
                setattr(parent, attr, new_mod)

            replaced.append(mod_path)
            continue

        # Case 4: GPT-Conv1D style (weight shape [out, in])
        if hasattr(old_mod, "weight") and tensor.dim() == 2 and old_mod.weight.dim() == 1:
            out_f, in_f = tensor.shape
            print(f"[rebuild] Conv1D-style reset {mod_path}: in={in_f}, out={out_f}")
            new_mod = nn.Linear(in_f, out_f)

            if isinstance(attr, int):
                parent[attr] = new_mod
            else:
                setattr(parent, attr, new_mod)

            replaced.append(mod_path)
            continue

        skipped.append(key)

    print("[rebuild] Replaced:", replaced[:20])
    print("[rebuild] Skipped:", skipped[:20])

    return student_model

# ------------------------------------------
# convenience wrappers (unchanged, small edits)
# ------------------------------------------

def regenaura_upscale_only(
    teacher_ckpt_path: str,
    student_model: nn.Module,
    ascend_cfg: RegenAscendConfig,
    output_dir: str = "./regen_output",
    run_name: str = "regen_upscale_only",
) -> RegenRunSummary:
    cfg = RegenConfig(
        ascend=ascend_cfg,
        distill=None,
        output_dir=output_dir,
        run_name=run_name,
        normalize_teacher=True,
        normalize_student_init=False,
    )
    return regenaura_from_checkpoint(
        teacher_ckpt_path=teacher_ckpt_path,
        student_model=student_model,
        regen_cfg=cfg,
        student_init_ckpt=None,
        tokenizer=None,
    )


def regenaura_distill_only(
    teacher_ckpt_path: str,
    student_model: nn.Module,
    distill_cfg: RegenDistillConfig,
    output_dir: str = "./regen_output",
    run_name: str = "regen_distill_only",
    *,
    student_init_ckpt: Optional[str] = None,
    tokenizer: Optional[Any] = None,
) -> RegenRunSummary:
    cfg = RegenConfig(
        ascend=None,
        distill=distill_cfg,
        output_dir=output_dir,
        run_name=run_name,
        normalize_teacher=True,
        normalize_student_init=False,
    )
    cfg.dataloader = distill_cfg.dataloader
    return regenaura_from_checkpoint(
        teacher_ckpt_path=teacher_ckpt_path,
        student_model=student_model,
        regen_cfg=cfg,
        student_init_ckpt=student_init_ckpt,
        tokenizer=tokenizer,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="REGENAURA v1.1 – patched orchestrator")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher checkpoint (.pt)")
    parser.add_argument("--mode", type=str, default="distill_only",
                        choices=["ascend_manual", "ascend_auto", "ascend_expand", "ascend_hybrid", "distill_only"])
    parser.add_argument("--output_dir", type=str, default="./regen_output")
    parser.add_argument("--run_name", type=str, default="regen_cli_run")
    args = parser.parse_args()

    print("[REGENAURA] CLI path is mainly for sanity-checking wiring.")
    print(f"[REGENAURA] teacher={args.teacher}, mode={args.mode}, output_dir={args.output_dir}, run_name={args.run_name}")
