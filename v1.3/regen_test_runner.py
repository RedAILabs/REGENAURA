import os
import torch
from regenaura import (
    regenaura_upscale_only,
    regenaura_distill_only,
    RegenConfig,
    RegenAscendConfig,
    RegenDistillConfig,
    apply_checkpoints_in_place,   # <<== NEW: import our sequential loader
)
from types import SimpleNamespace
import torch.nn as nn
from torch.serialization import add_safe_globals
from torch.cuda.amp.grad_scaler import GradScaler
add_safe_globals([GradScaler])

# ---- User edit: paths ----
CHECKPOINT_DIR = "./checkpoints"   # where your 70M and deepseek shards are
RED_70M_PATH = os.path.join(CHECKPOINT_DIR, "red_70m.pt")  # or .safetensors

# Example DeepSeek shards (you can rename if required)
DEEPSEEK_SHARD_1 = os.path.join(CHECKPOINT_DIR, "deepseek-00001-of-00002.safetensors")
DEEPSEEK_SHARD_2 = os.path.join(CHECKPOINT_DIR, "deepseek-00002-of-00002.safetensors")
DEEPSEEK_SHARDS = [DEEPSEEK_SHARD_1, DEEPSEEK_SHARD_2]

OUT_DIR = "./regen_run_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Basic device config ----
device_cpu = "cpu"
device_gpu = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TEST RUNNER] Detected device GPU? {'yes' if device_gpu == 'cuda' else 'no'}")

# ---- Minimal student model stub (REPLACE with your student model) ----
# The student_model must be an nn.Module and have a .config compatible with regenaura wrappers.
class SmallStudent(nn.Module):
    def __init__(self):
        super().__init__()
        vocab = 1000
        hidden = 128

        self.embed = nn.Embedding(vocab, hidden)
        self.l1 = nn.Linear(hidden, vocab)  # per-token output (like LM head)

        self.config = SimpleNamespace(d_model=hidden, num_layers=2)

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("expected input_ids")

        # input_ids: [batch, seq_len]
        x = self.embed(input_ids)  # [batch, seq_len, hidden]

        logits = self.l1(x)  # [batch, seq_len, vocab]

        return SimpleNamespace(logits=logits)

def build_student_model():
    # Replace with your actual student model constructor (T5-based Red)
    return SmallStudent()

# ---- Minimal dataloader for distill (tiny synthetic data) ----
from torch.utils.data import Dataset, DataLoader
class TinyTextDataset(Dataset):
    def __init__(self, size=100, seq_len=8, vocab=1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab = vocab
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        import torch
        input_ids = torch.randint(0, self.vocab, (self.seq_len,), dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

tiny_loader = DataLoader(TinyTextDataset(size=200), batch_size=1, shuffle=True, num_workers=0)

# ================================================================
# 1) Smoke test: load Red 70M and run upscale-only
# ================================================================
student = build_student_model()
cfg = RegenConfig(device=device_cpu)
cfg.dataloader = tiny_loader

asc_cfg = RegenAscendConfig(mode="ascend_expand", max_params=80_000_000)
print("[TEST] Running small upscale-only test with 70M checkpoint (fast)")
try:
    summary = regenaura_upscale_only(RED_70M_PATH, student, asc_cfg, output_dir=OUT_DIR, run_name="test_upscale")
    print("[TEST] Upscale done:", summary)
except Exception as e:
    print("[TEST] Upscale-only test failed (non-fatal):", e)

# ================================================================
# 2) LOAD DEEPSEEK SHARDS WITHOUT MERGING
# ================================================================
print("\n[TEST] Checking DeepSeek shards (no merge, sequential apply)...")

all_exist = all(os.path.exists(p) for p in DEEPSEEK_SHARDS)
if not all_exist:
    print("[TEST] One or more DeepSeek shards not found. Skipping.")
else:
    # Build a student model to apply weights into (for smoke test only)
    # Build a teacher model instance (same architecture as student)
    teacher = build_student_model()  # for real runs, use real T5 Red class
    print("[TEST] Applying DeepSeek shards into teacher model (no merge)...")
    try:
        res = apply_checkpoints_in_place(teacher, DEEPSEEK_SHARDS, device="cpu", overwrite=False, verbose=True)
        print("[TEST] Applied to teacher:", res)
        # Now we can call distill with teacher in place:
        # Option 1: pass teacher by path? Our regenaura expects teacher ckpt path, so save teacher
        teacher_path = os.path.join(OUT_DIR, "teacher_from_shards.pt")
        torch.save({"model_state_dict": teacher.state_dict()}, teacher_path)
        print("[TEST] Saved teacher snapshot to:", teacher_path)
    except Exception as e:
        print("[TEST] DeepSeek apply failed:", e)

# ================================================================
# 3) Distill-only short run (20 steps)
# ================================================================
print("\n[TEST] Running short distill-only test (20 steps) on student model (CPU/GPU per regen_cfg)")
dist_cfg = RegenDistillConfig(steps=20, temperature=2.0, alpha_kl=0.7, alpha_ce=0.3)

cfg.dataloader = tiny_loader
dist_cfg.dataloader = tiny_loader

try:
    summary = regenaura_distill_only(
        RED_70M_PATH,
        student,
        dist_cfg,
        output_dir=OUT_DIR,
        run_name="test_distill"
    )
    print("[TEST] Distill-only done:", summary)
except Exception as e:
    print("[TEST] Distill-only failed (non-fatal):", e)

print("\n[TEST] Runner finished.")
