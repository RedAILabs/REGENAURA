# File: rebuild_student_linears_from_state.py
# RegenAura v1.3 — Architecture Auto-Rebuilder
# -------------------------------------------
# Safely rebuild student model layers so that expanded state dicts can load
# without shape mismatch errors. Works for linear layers, embeddings, and
# output projection layers. Nonlinear / custom modules are passed through.

import torch
import torch.nn as nn


def _resize_linear(layer: nn.Linear, new_weight: torch.Tensor, new_bias: torch.Tensor):
    """Resize a Linear layer to match new tensor shapes."""
    in_features = new_weight.size(1)
    out_features = new_weight.size(0)

    new_layer = nn.Linear(in_features, out_features, bias=new_bias is not None)
    with torch.no_grad():
        new_layer.weight.copy_(new_weight)
        if new_bias is not None:
            new_layer.bias.copy_(new_bias)
    return new_layer


def _resize_embedding(layer: nn.Embedding, new_weight: torch.Tensor):
    """Resize embedding to match new tensor shape."""
    num_embeddings, embedding_dim = new_weight.shape
    new_layer = nn.Embedding(num_embeddings, embedding_dim)
    with torch.no_grad():
        new_layer.weight.copy_(new_weight)
    return new_layer


def _resize_layer_norm(layer: nn.LayerNorm, new_weight: torch.Tensor, new_bias: torch.Tensor):
    """Resize LayerNorm to match new hidden size."""
    hidden_size = new_weight.size(0)
    new_layer = nn.LayerNorm(hidden_size, elementwise_affine=True)
    with torch.no_grad():
        new_layer.weight.copy_(new_weight)
        new_layer.bias.copy_(new_bias)
    return new_layer


def rebuild_student_linears_from_state(model: nn.Module, expanded_state: dict):
    """
    Rebuilds Linear, Embedding, and LayerNorm modules to match the shapes
    inside expanded_state so state_dict can load cleanly.

    This is the heavy-lifter that enables ascend_expand → rebuild → load pipeline.
    """

    model_sd = model.state_dict()

    # --- PASS 1: Collect mismatched keys ---
    mismatches = []
    for k, v in expanded_state.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            mismatches.append((k, model_sd[k].shape, v.shape))

    if len(mismatches) == 0:
        print("[REBUILD] No shape mismatches detected.")
        return model

    print(f"[REBUILD] Detected {len(mismatches)} mismatched layers. Rebuilding...")

    # --- PASS 2: Rebuild modules ---
    for key, old_shape, new_shape in mismatches:

        module_name, param_name = key.rsplit(".", 1)

        # Walk the model to the module
        mod = model
        parts = module_name.split(".")
        for p in parts:
            if not hasattr(mod, p):
                mod = None
                break
            mod = getattr(mod, p)

        if mod is None:
            print(f"[REBUILD][WARN] Could not find module for key {key}")
            continue

        # Identify module types
        if isinstance(mod, nn.Linear):
            if param_name == "weight":
                new_weight = expanded_state[key]
                new_bias = expanded_state.get(module_name + ".bias", None)
                rebuilt = _resize_linear(mod, new_weight, new_bias)
                setattr_recursive(model, module_name, rebuilt)

        elif isinstance(mod, nn.Embedding):
            if param_name == "weight":
                new_emb = expanded_state[key]
                rebuilt = _resize_embedding(mod, new_emb)
                setattr_recursive(model, module_name, rebuilt)

        elif isinstance(mod, nn.LayerNorm):
            if param_name == "weight":
                new_weight = expanded_state[key]
                new_bias = expanded_state.get(module_name + ".bias", None)
                rebuilt = _resize_layer_norm(mod, new_weight, new_bias)
                setattr_recursive(model, module_name, rebuilt)

        else:
            print(f"[REBUILD][SKIP] Module '{module_name}' type {type(mod)} not supported.")

    print("[REBUILD] Rebuild complete.")
    return model


def setattr_recursive(model: nn.Module, path: str, new_mod: nn.Module):
    """Assign a new module deep inside nested modules (model.layer1.attn.q_proj etc)."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)
