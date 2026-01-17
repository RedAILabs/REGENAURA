# REGENAURA

REGENAURA is a lightweight, production-oriented pretraining framework designed for efficient large language model development. REGENAURA focuses on correctness, stability, and performance rather than flashy abstractions.

## What REGENAURA Achieves

* **Streaming pretraining pipeline** with near-zero RAM overhead
* **T5-style span corruption** with realistic loss behavior
* **Honest loss computation** (no label leakage, no artificial loss collapse)
* **Stable BF16 training** with gradient checkpointing support
* **Robust checkpointing** (model, optimizer, scheduler, global step)
* **Warmup + cosine LR scheduling** with correct resume behavior
* **Fast startup & training** without offline dataset preprocessing

## Key Design Principles

* Correctness over speed hacks
* Memory efficiency by default
* Training realism over misleading metrics
* Minimal but extensible codebase

## Current Capabilities

* Supports encoder–decoder pretraining (T5-style)
* Handles large text corpora via streaming datasets
* Scales cleanly from local experiments to long-running training jobs
* Designed for research and controlled production environments

REGENAURA represents a stable foundation for serious model training workflows where reliability and transparency matter more than shortcuts.
