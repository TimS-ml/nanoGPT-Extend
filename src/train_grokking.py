#!/usr/bin/env python3
"""
Grokking experiment: modular arithmetic with a tiny transformer.

Reproduces the grokking phenomenon where:
  - Training accuracy reaches 100% quickly (~step 300)
  - Validation accuracy suddenly jumps to 100% much later (~step 3000-5000)

Config follows reference/nanuGPT/configs/grokking/prime223.yaml:
  - prime=223, operation=x+y, weight_decay=0.1, lr=1e-3
  - Fast grokking (~10 minutes on a single GPU)

Usage:
    python train_grokking.py [--prime 223] [--operation x+y] [--max_steps 10000]

Adapted from reference/nanuGPT (grokking_data.py, grokking_loss.py, grokking_tokenizer.py, tiny_transformer.py)
"""

import os
import sys
import json
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ============================================================================
# Grokking Tokenizer (inline)
# ============================================================================

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x * y % p, y, x),
}

ALL_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS,
    "x^2+y^2": lambda x, y, _: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, _: (x, y, x**2 + x * y + y**2),
    "x^3+xy": lambda x, y, _: (x, y, x**3 + x * y),
}


class GrokkingTokenizer:
    def __init__(self, prime, operations):
        self.prime = prime
        self.eos_token = "<|eos|>"
        self.eq_token = "="
        self.all_tokens = (
            [self.eos_token, self.eq_token]
            + list(sorted(operations))
            + list(range(prime))
        )
        self.vocab_size = len(self.all_tokens)
        self.token_to_idx = {t: i for i, t in enumerate(self.all_tokens)}

    def __getitem__(self, token):
        return self.token_to_idx[token]


# ============================================================================
# Grokking Data Generation
# ============================================================================


def make_grokking_data(operation, prime, tokenizer):
    """Generate all (a op b) mod p equations as token sequences."""
    eos = tokenizer[tokenizer.eos_token]
    eq = tokenizer[tokenizer.eq_token]
    op = tokenizer[operation]

    b_start = 1 if operation in DIVISION_MODULO_OPERATIONS else 0
    equations = torch.cartesian_prod(torch.arange(prime), torch.arange(b_start, prime))

    result = ALL_OPERATIONS[operation](equations[:, 0], equations[:, 1], prime)
    equations = torch.stack((result[0], result[1], result[2] % prime), dim=1)

    # Map numbers to token ids
    token_map = torch.tensor(
        [tokenizer[i] for i in range(torch.max(equations) + 1)], dtype=torch.int32
    )
    equations = token_map[equations]

    # Labels: the result column (for classification)
    labels = equations[:, 2].to(torch.int64)

    # Input: [EOS, a, op, b, =]
    a_col = equations[:, 0:1]
    b_col = equations[:, 1:2]
    n = equations.size(0)
    inputs = torch.cat(
        [
            torch.full((n, 1), eos),
            a_col,
            torch.full((n, 1), op),
            b_col,
            torch.full((n, 1), eq),
        ],
        dim=1,
    ).to(torch.long)

    return inputs, labels


# ============================================================================
# Tiny Transformer (no einops dependency)
# ============================================================================


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            n_embd, n_heads, dropout=dropout, bias=True
        )
        self.self_attn_norm = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x: (seq_len, batch_size, n_embd)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[0], device=x.device
        )
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)
        return a2


class TinyTransformer(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(context_len, n_embd)
        self.layers = nn.ModuleList(
            [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, inputs):
        # inputs: (batch_size, context_len)
        B, T = inputs.shape
        tok_emb = self.token_embeddings(inputs)  # (B, T, D)
        pos = torch.arange(T, device=inputs.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embeddings(pos)  # (B, T, D)
        x = (tok_emb + pos_emb).permute(1, 0, 2)  # (T, B, D) for MultiheadAttention
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (T, B, vocab_size)
        return logits


# ============================================================================
# Training
# ============================================================================


def get_grokking_loss(logits, labels):
    """Last-token classification loss for grokking."""
    last_logits = logits[-1, :, :]  # (batch_size, vocab_size)
    loss = F.cross_entropy(last_logits, labels)
    correct = (torch.argmax(last_logits, dim=-1) == labels).sum().item()
    return loss, correct


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss, correct = get_grokking_loss(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            total_correct += correct
            total_samples += inputs.size(0)
    model.train()
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description="Grokking experiment")
    parser.add_argument("--prime", type=int, default=223)
    parser.add_argument("--operation", type=str, default="x+y")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="out/grokking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = args.device
    print(f"Device: {device}")
    print(f"Experiment: {args.operation} mod {args.prime}, wd={args.weight_decay}")

    # ---- Data ----
    tokenizer = GrokkingTokenizer(args.prime, list(ALL_OPERATIONS.keys()))
    inputs, labels = make_grokking_data(args.operation, args.prime, tokenizer)
    context_len = inputs.shape[1]  # 5 tokens: [EOS, a, op, b, =]

    dataset = TensorDataset(inputs, labels)
    train_size = int(args.training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, train_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(args.batch_size, val_size), shuffle=False
    )

    print(f"Data: {len(dataset)} total, {train_size} train, {val_size} val")
    print(f"Vocab size: {tokenizer.vocab_size}, Context length: {context_len}")

    # ---- Model ----
    model = TinyTransformer(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        vocab_size=tokenizer.vocab_size,
        context_len=context_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e3:.1f}K parameters")

    # ---- Optimizer (AdamW for grokking, not Muon) ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    # ---- Training loop ----
    metrics_log = []
    best_val_acc = 0.0
    step = 0
    train_iter = iter(train_loader)

    print(f"\nStarting training for {args.max_steps} steps...")
    t0 = time.time()

    model.train()
    while step < args.max_steps:
        # Get batch (cycle through data)
        try:
            batch_inputs, batch_labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_inputs, batch_labels = next(train_iter)

        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # Forward
        logits = model(batch_inputs)
        loss, correct = get_grokking_loss(logits, batch_labels)
        train_acc = correct / batch_inputs.size(0)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1

        # Log
        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(
                f"step {step:6d} | train_loss {loss.item():.4f} | train_acc {train_acc:.4f} | time {dt:.1f}s"
            )

        # Evaluate
        if step % args.eval_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            dt = time.time() - t0
            print(
                f"  [EVAL] step {step:6d} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | time {dt:.1f}s"
            )

            entry = {
                "step": step,
                "train_loss": loss.item(),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time": dt,
            }
            metrics_log.append(entry)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    "model": model.state_dict(),
                    "step": step,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": vars(args),
                }
                torch.save(checkpoint, os.path.join(args.out_dir, "ckpt_best.pt"))
                print(f"  >> New best val_acc: {val_acc:.4f}, saved checkpoint")

            # Early stopping if fully grokked
            if val_acc >= 0.99 and train_acc >= 0.99:
                print(f"\n*** GROKKING ACHIEVED at step {step}! ***")
                print(f"    train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
                break

    # ---- Save final results ----
    total_time = time.time() - t0
    print(f"\nTraining finished in {total_time:.1f}s ({step} steps)")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    # Save final checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "step": step,
        "val_acc": best_val_acc,
        "config": vars(args),
    }
    torch.save(checkpoint, os.path.join(args.out_dir, "ckpt_final.pt"))

    # Save metrics
    results = {
        "config": vars(args),
        "total_time": total_time,
        "total_steps": step,
        "best_val_acc": best_val_acc,
        "metrics": metrics_log,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {args.out_dir}/metrics.json")


if __name__ == "__main__":
    main()
