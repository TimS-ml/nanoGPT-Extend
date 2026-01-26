"""
Training script for Encoder-Decoder Transformer (translation tasks).

This is the seq2seq equivalent of train.py for decoder-only models.

Usage:
    python src/train_translation.py config/train_translation.py

    # With overrides:
    python src/train_translation.py config/train_translation.py --device=cpu
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
from os.path import exists

import torch
from torch.optim.lr_scheduler import LambdaLR

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import make_model, greedy_decode
from src.shared.lr_scheduler import rate
from src.shared.training import TrainState, Batch, run_epoch, DummyOptimizer, DummyScheduler
from src.shared.loss import LabelSmoothing, SimpleLossCompute
from src.seq2seq_data import (
    data_gen,
    create_synthetic_dataloaders,
    SPACY_AVAILABLE,
    TORCHTEXT_AVAILABLE,
)

# -----------------------------------------------------------------------------
# Default configuration values
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out-translation'
eval_interval = 500
log_interval = 10
eval_iters = 100
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'translation'
wandb_run_name = 'transformer'

# Data
dataset = 'synthetic'  # 'synthetic' or 'multi30k'
language_pair = ('de', 'en')
max_padding = 72

# Model
N = 6           # Number of layers (n_layer)
d_model = 512   # Model dimension (n_embd)
d_ff = 2048     # Feed-forward dimension
h = 8           # Number of heads (n_head)
dropout = 0.1

# Optimizer
learning_rate = 1.0
warmup = 3000
beta1 = 0.9
beta2 = 0.98
eps = 1e-9
label_smoothing = 0.1

# Training
batch_size = 32
gradient_accumulation_steps = 10
num_epochs = 8
max_iters = None

# Distributed
distributed = False

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False  # Use torch.compile (PyTorch 2.0+)
dtype = 'float32'

# Checkpoint
file_prefix = 'model_'

# -----------------------------------------------------------------------------
# Load configuration
# -----------------------------------------------------------------------------

# Parse command line arguments for config file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, tuple))]
exec(open('src/configurator.py').read()) if exists('src/configurator.py') else None

# Override from command line (simple parsing)
for arg in sys.argv[1:]:
    if '=' in arg:
        key, val = arg.split('=', 1)
        key = key.lstrip('-')
        if key in config_keys:
            try:
                # Try to evaluate as Python literal
                globals()[key] = eval(val)
            except:
                globals()[key] = val

# Load config file if provided
if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
    config_file = sys.argv[1]
    print(f"Loading config from {config_file}")
    with open(config_file) as f:
        exec(f.read())

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# wandb logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        'N': N, 'd_model': d_model, 'd_ff': d_ff, 'h': h,
        'dropout': dropout, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'warmup': warmup,
    })

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

print(f"Dataset: {dataset}")

if dataset == 'synthetic':
    # Use synthetic copy task for testing
    V = 11  # Vocabulary size
    src_vocab_size = V
    tgt_vocab_size = V
    pad_idx = 0

    def get_train_batches():
        return data_gen(V, batch_size, nbatches=20, device=device)

    def get_valid_batches():
        return data_gen(V, batch_size, nbatches=5, device=device)

    print(f"Using synthetic copy task with vocab_size={V}")

elif dataset == 'multi30k':
    # Real translation dataset
    if not (SPACY_AVAILABLE and TORCHTEXT_AVAILABLE):
        raise ImportError(
            "multi30k dataset requires spacy and torchtext. "
            "Install with: pip install spacy torchtext && "
            "python -m spacy download de_core_news_sm en_core_web_sm"
        )

    from src.seq2seq_data import load_tokenizers, load_vocab, create_dataloaders

    print("Loading tokenizers...")
    spacy_src, spacy_tgt = load_tokenizers(language_pair[0], language_pair[1])

    print("Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_src, spacy_tgt, vocab_path=f"{out_dir}/vocab.pt")

    src_vocab_size = len(vocab_src)
    tgt_vocab_size = len(vocab_tgt)
    pad_idx = vocab_tgt["<blank>"]

    print("Creating dataloaders...")
    train_dataloader, valid_dataloader = create_dataloaders(
        device=device,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_src=spacy_src,
        spacy_tgt=spacy_tgt,
        batch_size=batch_size,
        max_padding=max_padding,
        is_distributed=distributed,
    )

    def get_train_batches():
        return (Batch(b[0], b[1], pad_idx) for b in train_dataloader)

    def get_valid_batches():
        return (Batch(b[0], b[1], pad_idx) for b in valid_dataloader)

else:
    raise ValueError(f"Unknown dataset: {dataset}")

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------

print(f"Initializing model: N={N}, d_model={d_model}, h={h}, d_ff={d_ff}")

model = make_model(
    src_vocab=src_vocab_size,
    tgt_vocab=tgt_vocab_size,
    N=N,
    d_model=d_model,
    d_ff=d_ff,
    h=h,
    dropout=dropout,
)
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params:,}")

# Compile model if requested (PyTorch 2.0+)
if compile and hasattr(torch, 'compile'):
    print("Compiling model...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Optimizer and scheduler
# -----------------------------------------------------------------------------

criterion = LabelSmoothing(
    size=tgt_vocab_size,
    padding_idx=pad_idx,
    smoothing=label_smoothing,
)
criterion.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=eps,
)

lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=warmup),
)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

print(f"\nStarting training for {num_epochs} epochs...")
print(f"Device: {device}, dtype: {dtype}")
print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
print("-" * 70)

train_state = TrainState()
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    print(f"\nEpoch {epoch + 1}/{num_epochs} - Training")

    with ctx:
        train_loss, train_state = run_epoch(
            get_train_batches(),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=gradient_accumulation_steps,
            train_state=train_state,
        )

    print(f"Training loss: {train_loss:.4f}")

    # Validation
    model.eval()
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation")

    with torch.no_grad():
        with ctx:
            val_loss, _ = run_epoch(
                get_valid_batches(),
                model,
                SimpleLossCompute(model.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )

    print(f"Validation loss: {val_loss:.4f}")

    # wandb logging
    if wandb_log:
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
        })

    # Save checkpoint
    if val_loss < best_val_loss or always_save_checkpoint:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(out_dir, f"{file_prefix}{epoch:02d}.pt")
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)

# Save final model
final_path = os.path.join(out_dir, f"{file_prefix}final.pt")
print(f"\nSaving final model to {final_path}")
torch.save(model.state_dict(), final_path)

# -----------------------------------------------------------------------------
# Test inference
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("Testing inference with greedy decoding...")

model.eval()

if dataset == 'synthetic':
    # Test on copy task
    test_src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)
    test_src_mask = torch.ones(1, 1, test_src.size(1)).to(device)

    with torch.no_grad():
        output = greedy_decode(
            model, test_src, test_src_mask,
            max_len=test_src.size(1),
            start_symbol=1,
        )

    print(f"Input:  {test_src[0].tolist()}")
    print(f"Output: {output[0].tolist()}")

print("\nTraining complete!")
