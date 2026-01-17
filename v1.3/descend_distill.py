# File: descend_distill.py
# RegenAura v1.3 — Stable Soft Distillation Engine
# -------------------------------------------------
# This module provides clean, deterministic soft-distillation between
# teacher and student models with temperature scaling + KL + CE losses.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler


# ---------------------------------------------------------
# Deterministic Environment
# ---------------------------------------------------------

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------

def _kl_loss(student_logits, teacher_logits, temperature):
    """
    KL divergence between softened teacher distribution and student.
    """
    t = temperature
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean") * (t * t)


def _ce_loss(student_logits, teacher_logits):
    """
    Cross-Entropy loss using teacher's argmax as labels.
    """
    with torch.no_grad():
        targets = teacher_logits.argmax(dim=-1)
    return F.cross_entropy(student_logits, targets)


# ---------------------------------------------------------
# Main soft distillation function
# ---------------------------------------------------------

def descend_soft_distill(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader,
    *,
    device="cuda",
    steps=200,
    temperature=2.0,
    alpha_kl=0.7,
    alpha_ce=0.3,
    max_grad_norm=1.0,
    use_amp=True,
    precision="bf16",
):
    """
    Perform soft distillation using the teacher's logits as targets.
    Compatible with RegenAura orchestrator.

    teacher_model: model producing logits
    student_model: model to train
    dataloader   : yields batches {"input_ids", "attention_mask"}
    """
    set_global_seed(42)

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=use_amp)

    step = 0
    data_iter = iter(dataloader)

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp, dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
            # -----------------------
            # Teacher forward
            # -----------------------
            with torch.no_grad():
                t_out = teacher_model(input_ids=input_ids, attention_mask=attn)
                teacher_logits = t_out.logits

            # -----------------------
            # Student forward
            # -----------------------
            s_out = student_model(input_ids=input_ids, attention_mask=attn)
            student_logits = s_out.logits

            # -----------------------
            # Compute losses
            # -----------------------
            kl = _kl_loss(student_logits, teacher_logits, temperature)
            ce = _ce_loss(student_logits, teacher_logits)
            total_loss = alpha_kl * kl + alpha_ce * ce

        # -----------------------
        # Backward (AMP or FP32)
        # -----------------------
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # -----------------------
        # Logging
        # -----------------------
        if step % 10 == 0:
            print(
                f"[DESCEND] step {step}/{steps} "
                f"loss={total_loss.item():.4f} "
                f"kl={kl.item():.4f} "
                f"ce={ce.item():.4f}"
            )

        step += 1

    return student_model
