# Muon Optimizer + Grokking Training Plan

**Branch:** `training-muon-grokking`
**Base:** Karpathy's nanoGPT (`src/`)
**Hardware:** Laptop 4090 (16GB VRAM) + 512GB storage

> **From now on, 可以交给子agent的步骤都使用 `opencode run "<prompt>"`.**

---

## Overview

Two experiments on one branch:

| Experiment | Model | Data | Goal |
|---|---|---|---|
| **A. Grokking** | Tiny Transformer (~457K params) | Synthetic modular arithmetic | Reproduce grokking phenomenon: train acc=100% early, val acc jumps to 100% much later |
| **B. GPT LM Training** | GPT-2 Small variant (~124M params, Keller-style) | FineWeb-Edu-10B (or OpenWebText) | Train a language model with Muon optimizer, compare vs AdamW baseline |

---

## Architecture Decisions

### Optimizer: Muon + Aux Adam (from nanuGPT reference)
- **Muon** for 2D hidden weight matrices (attention QKVO, MLP weights)
  - Newton-Schulz orthogonalization (5 iterations, bfloat16)
  - Momentum 0.85→0.95 warmup over 300 steps
  - LR: 0.05
- **AdamW** for everything else:
  - Head params: lr=0.008, betas=(0.8, 0.95)
  - Embeddings: lr=0.6, betas=(0.8, 0.95)
  - Scalars (biases, norms): lr=0.04, betas=(0.8, 0.95)
- Source: `reference/nanuGPT/nanugpt/optimizers/muon_optim.py` → `SingleDeviceMuonWithAuxAdam`

### Model (GPT LM): Keller-style GPT
- RMSNorm (no learnable params) instead of LayerNorm
- RoPE instead of absolute positional embeddings
- No bias anywhere
- No weight tying (required for Muon's per-group LR)
- Scaled residual: `1/sqrt(2*n_layer)`
- Source: `reference/nanuGPT/nanugpt/models/nanogpt_keller.py`

### Model (Grokking): Tiny Transformer
- 2 layers, 128 dim, 4 heads
- Post-norm, absolute positional embeddings
- `nn.MultiheadAttention` from PyTorch
- Source: `reference/nanuGPT/nanugpt/models/tiny_transformer.py`

### Precision: bfloat16
- 4090 has native bf16 support
- No GradScaler needed (unlike fp16)
- fp8 rejected: requires custom Triton kernels + multi-GPU infrastructure from modded-nanogpt

### LR Schedule: WSD (Warmup-Stable-Decay)
- Linear warmup → constant → linear cooldown (last 40% of training)
- Better than cosine for Muon based on nanuGPT benchmarks

---

## File Structure (new files to create)

```
src/
├── train_muon.py          # Main GPT training script with Muon
├── train_grokking.py      # Grokking experiment script
├── muon_optim.py          # Muon optimizer (adapted from nanuGPT, single-GPU)
├── model_keller.py        # Keller-style GPT (RMSNorm, RoPE, no bias)
└── model_tiny.py          # Tiny transformer for grokking

data/
├── fineweb_edu/            # FineWeb-Edu tokenized shards
│   └── prepare.py          # Download + tokenize script
└── shakespeare_char/       # Already exists (fallback for testing)

out/
├── muon_gpt/              # GPT training outputs
│   ├── ckpt.pt            # Model checkpoint (bf16 weights)
│   └── metrics.json       # Training metrics (loss, lr, etc.)
└── grokking/              # Grokking outputs
    ├── ckpt.pt
    └── metrics.json
```

---

## Step-by-Step Execution Plan

### Phase 0: Environment Setup
```bash
# Activate env
mamba activate llm   # or: micromamba activate /home/tim/miniforge3/envs/llm

# Fix PyTorch (currently broken in llm env)
pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Install missing deps
pip install tiktoken datasets tqdm
```
**Status:** ❌ PyTorch install interrupted, needs retry

Use `opencode run`:
```
opencode run "Fix PyTorch in the llm conda env at /home/tim/miniforge3/envs/llm. 
The current torch 2.8.0 is broken (AttributeError: module 'torch._C' has no attribute '_SafeKernelFunction'). 
Steps: 
1. micromamba activate /home/tim/miniforge3/envs/llm 
2. pip uninstall torch -y 
3. pip install torch --index-url https://download.pytorch.org/whl/cu126 
4. Verify: python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
5. Also install: pip install tiktoken datasets tqdm numpy
6. Verify all imports work"
```

### Phase 1: Write Code (no GPU needed)

**1a. `src/muon_optim.py`** — Muon optimizer
- Copy `SingleDeviceMuonWithAuxAdam` from nanuGPT
- Strip DDP variants (single GPU only)
- Keep: `zeropower_via_newtonschulz5`, `muon_update`, `adam_update`
- Add: momentum warmup logic

**1b. `src/model_keller.py`** — Keller GPT model
- Adapt from `reference/nanuGPT/nanugpt/models/nanogpt_keller.py`
- Remove nanuGPT dependency (`from nanugpt import common`)
- Self-contained: Rotary, RMSNorm, CausalSelfAttention, MLP, Block, GPT
- Add `configure_muon_optimizer()` method

**1c. `src/model_tiny.py`** — Tiny transformer for grokking
- Adapt from `reference/nanuGPT/nanugpt/models/tiny_transformer.py`
- Remove `einops` dependency (use native torch ops)
- Remove nanuGPT dependency

**1d. `src/train_grokking.py`** — Grokking training script
- Self-contained (no YAML config system)
- Inline grokking data generation (modular arithmetic)
- Inline grokking tokenizer
- AdamW optimizer (grokking uses AdamW, not Muon)
- Config: prime=223, weight_decay=0.1, lr=1e-3, max_steps=10000
- Logs: train_loss, val_loss, train_acc, val_acc per step
- Save metrics to JSON + checkpoint

**1e. `src/train_muon.py`** — GPT training with Muon
- Based on `src/train.py` but with:
  - Muon optimizer instead of AdamW
  - Keller GPT model instead of vanilla GPT
  - WSD lr schedule
  - bfloat16 autocast
  - Memory-optimized for 16GB VRAM
- Config for 4090:
  - batch_size=8, block_size=1024, grad_accum_steps=8
  - Effective batch: 8*8*1024 = 65K tokens/iter
  - n_layer=12, n_head=12, n_embd=768 (~124M params)

**1f. `data/fineweb_edu/prepare.py`** — Data preparation
- Download FineWeb-Edu-10B sample via HuggingFace datasets
- Tokenize with tiktoken (GPT-2 BPE)
- Save as uint16 .bin shards

### Phase 2: Data Download

Use `opencode run`:
```
opencode run "Activate the llm env (micromamba activate /home/tim/miniforge3/envs/llm), 
then run: cd /home/tim/offline-git/GPT/mini/nanoGPT/data/fineweb_edu && python prepare.py --num_tokens 1e8
This downloads ~100M tokens of FineWeb-Edu and saves as train.bin/val.bin.
Wait for completion and report the file sizes."
```

### Phase 3: Run Grokking Experiment (~10 min)

Use `opencode run`:
```
opencode run "Activate the llm env (micromamba activate /home/tim/miniforge3/envs/llm), 
then run: cd /home/tim/offline-git/GPT/mini/nanoGPT/src && python train_grokking.py
This runs the grokking experiment (modular arithmetic, prime=223, ~10 min).
Expected: train acc=100% early (~step 300), val acc jumps to 100% later (~step 3000-5000).
Wait for it to complete. Report:
1. When did train_acc first reach 100%?
2. When did val_acc first reach 99%+? (= grokking point)
3. Total training time
4. Check that out/grokking/metrics.json and ckpt_best.pt were saved"
```

### Phase 4: Run GPT Training (~hours)

Use `opencode run`:
```
opencode run "Activate the llm env (micromamba activate /home/tim/miniforge3/envs/llm), 
then run: cd /home/tim/offline-git/GPT/mini/nanoGPT/src && python train_muon.py --dataset shakespeare_char --max_iters 5000 --eval_interval 250
This does a quick test run on Shakespeare data (~5000 steps).
If it works (no OOM, loss decreasing), then run the full training:
  python train_muon.py --max_iters 20000
Report: final val_loss, training time, any errors encountered.
Check that out/muon_gpt/metrics.json and ckpt_best.pt were saved."
```

### Phase 5: Verify & Commit

Use `opencode run`:
```
opencode run "In /home/tim/offline-git/GPT/mini/nanoGPT on branch training-muon-grokking:
1. git add src/muon_optim.py src/model_keller.py src/train_grokking.py src/train_muon.py data/fineweb_edu/prepare.py TRAINING_PLAN.md
2. git status to verify
3. git commit -m 'Add Muon optimizer + grokking experiment + GPT training scripts'
Do NOT push."
```

---

## Memory Budget (4090 16GB)

| Component | Est. Memory |
|---|---|
| Model params (124M × bf16) | ~248 MB |
| Muon state (momentum buffers) | ~500 MB |
| Adam state (embed+head+scalars) | ~500 MB |
| Activations (batch=8, seq=1024) | ~4-6 GB |
| Gradients | ~500 MB |
| **Total est.** | **~6-8 GB** |
| Headroom for peaks | ~8-10 GB |

If OOM: reduce batch_size to 4, increase grad_accum_steps to 16.

---

## Key References

- `reference/modded-nanogpt/train_gpt.py` — State-of-art NorMuon implementation (too complex, 8xH100 oriented)
- `reference/nanuGPT/nanugpt/optimizers/muon_optim.py` — Clean Muon implementation (we use this)
- `reference/nanuGPT/nanugpt/models/nanogpt_keller.py` — Keller GPT model (we adapt this)
- `reference/nanuGPT/configs/grokking/prime223.yaml` — Grokking config (fast, ~10 min)
- Muon paper: https://kellerjordan.github.io/posts/muon/
- Grokking paper: https://arxiv.org/abs/2201.02177
