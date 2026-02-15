#!/usr/bin/env python3
"""
GPT training with Muon optimizer, optimized for a single laptop 4090 (16GB VRAM).

Based on src/train.py but with:
  - Muon optimizer (for 2D hidden matrices) + AdamW (for embed/head/scalars)
  - Keller-style GPT model (RMSNorm, RoPE, no bias, no weight tying)
  - WSD learning rate schedule (warmup-stable-decay)
  - bfloat16 autocast (native 4090 support, no GradScaler needed)
  - Memory-optimized for 16GB VRAM

Usage:
    python train_muon.py                          # defaults (FineWeb-Edu)
    python train_muon.py --dataset shakespeare_char  # quick test
    python train_muon.py --max_iters 5000 --eval_interval 250

References:
  - reference/nanuGPT/configs/train_gpt2/openwebtext_tokens10b_keller_muon.yaml
  - reference/modded-nanogpt/train_gpt.py
"""

import os
import sys
import time
import json
import math
import argparse
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model_keller import GPT, GPTConfig
from muon_optim import configure_muon_optimizer


# ============================================================================
# LR Schedule: WSD (Warmup-Stable-Decay)
# ============================================================================


def get_lr_wsd(
    step, max_steps, max_lr, warmup_steps=256, cooldown_frac=0.4, end_factor=1e-2
):
    """
    Warmup-Stable-Decay schedule:
      1. Linear warmup for warmup_steps
      2. Constant at max_lr
      3. Linear cooldown over last cooldown_frac of training to max_lr * end_factor
    """
    if step < warmup_steps:
        return max_lr * step / max_steps  # will be multiplied by per-group lr
    cooldown_start = int(max_steps * (1 - cooldown_frac))
    if step >= cooldown_start:
        progress = (step - cooldown_start) / (max_steps - cooldown_start)
        return max_lr * (1 - progress * (1 - end_factor))
    return max_lr


def get_lr_multiplier_wsd(
    step, max_steps, warmup_steps=256, cooldown_frac=0.4, end_factor=1e-2
):
    """Returns a multiplier [0, 1] for the LR at a given step."""
    if step < warmup_steps:
        return step / warmup_steps
    cooldown_start = int(max_steps * (1 - cooldown_frac))
    if step >= cooldown_start:
        progress = (step - cooldown_start) / max(max_steps - cooldown_start, 1)
        return 1.0 - progress * (1 - end_factor)
    return 1.0


# ============================================================================
# Data Loading
# ============================================================================


def get_batch(data, batch_size, block_size, device):
    """Poor man's data loader: random batch from memmap data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = (
        x.pin_memory().to(device, non_blocking=True),
        y.pin_memory().to(device, non_blocking=True),
    )
    return x, y


@torch.no_grad()
def estimate_loss(
    model, train_data, val_data, eval_iters, batch_size, block_size, device, ctx
):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="GPT training with Muon optimizer")

    # I/O
    parser.add_argument("--out_dir", type=str, default="out/muon_gpt")
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=100)

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb_edu",
        help="Dataset directory under data/",
    )

    # Model
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--block_size", type=int, default=1024)

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Micro batch size per step (fits in 16GB VRAM)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Effective batch = batch_size * grad_accum * block_size tokens",
    )
    parser.add_argument("--max_iters", type=int, default=20000)

    # Muon optimizer
    parser.add_argument("--muon_lr", type=float, default=0.05)
    parser.add_argument("--head_lr", type=float, default=0.008)
    parser.add_argument("--embed_lr", type=float, default=0.6)
    parser.add_argument("--scalar_lr", type=float, default=0.04)

    # LR schedule (WSD)
    parser.add_argument("--warmup_steps", type=int, default=256)
    parser.add_argument("--cooldown_frac", type=float, default=0.4)
    parser.add_argument("--lr_end_factor", type=float, default=1e-2)

    # Gradient clipping
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Per-parameter gradient norm clip (Keller-style)",
    )

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--compile", action="store_true", default=True, help="Use torch.compile"
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    if args.no_compile:
        args.compile = False

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = args.device
    dtype = torch.bfloat16  # 4090 native support, no GradScaler needed
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if "cuda" in device
        else nullcontext()
    )

    tokens_per_iter = (
        args.gradient_accumulation_steps * args.batch_size * args.block_size
    )
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(
        f"Effective batch size: {args.gradient_accumulation_steps * args.batch_size} sequences"
    )

    # ---- Data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", args.dataset)
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print(f"ERROR: Training data not found at {data_dir}/train.bin")
        print(f"Run the data preparation script first:")
        print(f"  cd data/{args.dataset} && python prepare.py")
        sys.exit(1)

    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Val data:   {len(val_data):,} tokens")

    # Vocab size from meta.pkl or default GPT-2
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        print(f"Vocab size from meta.pkl: {vocab_size}")
    else:
        vocab_size = 50304
        print(f"Using default vocab size: {vocab_size}")

    # ---- Model ----
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)

    # ---- Optimizer ----
    optimizer = configure_muon_optimizer(
        model,
        muon_lr=args.muon_lr,
        head_lr=args.head_lr,
        embed_lr=args.embed_lr,
        scalar_lr=args.scalar_lr,
    )

    # ---- Compile ----
    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ---- Training Loop ----
    print(f"\nStarting training for {args.max_iters} iterations...")
    metrics_log = []
    best_val_loss = float("inf")
    iter_num = 0

    X, Y = get_batch(train_data, args.batch_size, args.block_size, device)
    t0 = time.time()
    running_mfu = -1.0

    while iter_num < args.max_iters:
        # LR schedule (WSD)
        lr_mult = get_lr_multiplier_wsd(
            iter_num,
            args.max_iters,
            args.warmup_steps,
            args.cooldown_frac,
            args.lr_end_factor,
        )
        for param_group in optimizer.param_groups:
            # Each group has its own base lr; multiply by schedule
            if "base_lr" not in param_group:
                param_group["base_lr"] = param_group["lr"]
            param_group["lr"] = param_group["base_lr"] * lr_mult

        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.eval_iters,
                args.batch_size,
                args.block_size,
                device,
                ctx,
            )
            print(
                f"step {iter_num:6d}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}"
            )

            entry = {
                "step": iter_num,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr_mult": lr_mult,
                "time": time.time() - t0,
            }
            metrics_log.append(entry)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                # Save checkpoint (bf16 weights to save space)
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                checkpoint = {
                    "model": {
                        k: v.to(torch.bfloat16)
                        for k, v in raw_model.state_dict().items()
                    },
                    "step": iter_num,
                    "val_loss": best_val_loss,
                    "train_loss": losses["train"],
                    "config": config.__dict__,
                    "args": vars(args),
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt_best.pt"))
                print(f"  >> New best val_loss: {best_val_loss:.4f}, saved checkpoint")

        # Gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            X, Y = get_batch(train_data, args.batch_size, args.block_size, device)
            loss.backward()

        # Gradient clipping (Keller-style: per-parameter norm)
        if args.grad_clip > 0:
            for p in model.parameters():
                if p.grad is not None:
                    g_norm = p.grad.norm()
                    if g_norm > 0:
                        p.grad.mul_(args.grad_clip / (g_norm + 1e-6))

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        if iter_num % args.log_interval == 0:
            lossf = loss.item() * args.gradient_accumulation_steps
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            if iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    args.batch_size * args.gradient_accumulation_steps,
                    dt / max(iter_num, 1),
                )
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {iter_num:6d}: loss {lossf:.4f}, lr_mult {lr_mult:.4f}, "
                f"time {dt:.1f}s, mfu {running_mfu * 100:.1f}%"
            )

        iter_num += 1

    # ---- Save final results ----
    total_time = time.time() - t0
    print(f"\nTraining finished in {total_time:.1f}s ({iter_num} steps)")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Save final checkpoint
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model": {k: v.to(torch.bfloat16) for k, v in raw_model.state_dict().items()},
        "step": iter_num,
        "val_loss": best_val_loss,
        "config": config.__dict__,
        "args": vars(args),
    }
    torch.save(checkpoint, os.path.join(args.out_dir, "ckpt_final.pt"))

    # Save metrics
    results = {
        "config": vars(args),
        "model_config": config.__dict__,
        "total_time": total_time,
        "total_steps": iter_num,
        "best_val_loss": best_val_loss,
        "tokens_per_iter": tokens_per_iter,
        "metrics": metrics_log,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
