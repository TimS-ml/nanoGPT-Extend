"""
BareGPT Training Script -- NumPy-only GPT training on Shakespeare.

Usage (from repo root):
    python -m src.bare_gpt.train              # Load weights or train, then generate
    python -m src.bare_gpt.train --train      # Force training even if weights exist
    python -m src.bare_gpt.train --epochs 500 # Override training steps

Original implementation by Damien Boureille, 2025 (MIT Licence).
"""

import argparse
import os
import pickle
import time

import numpy as np

from . import engine
from . import model
from . import utils

# Resolve paths relative to repo root (nanoGPT/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_DATA_PATH = os.path.join(_REPO_ROOT, "data", "shakespeare_char", "input.txt")
_OUT_DIR = os.path.join(_REPO_ROOT, "out", "bare_gpt")
_WEIGHTS_PATH = os.path.join(_OUT_DIR, "weights.pkl")

####  HYPERPARAMETERS  ####

d_model = 256
n_layers = 2
n_heads = 4
batch_size = 16
block_size = 128
epochs = 700
learning_rate = 0.25e-3

############################


def prepare_input(path):
    with open(path, "r") as file:
        text = file.read()

    # Strip useless whitespaces
    return text.strip()


def tokenize_data(data):
    vocab = sorted(list(set(data)))
    ids = utils.encode(data, vocab)
    return ids, vocab


def train(encoded_data, params, epochs, learning_rate):
    """
    Training loop
    Functions flow:
    1. Forward Pass:
        get_batch -> embed_tokens_and_positions -> forward -> transformer_block (N times)
        -> layernorm_forward -> multi_head_attention -> mlp_forward -> cross_entropy_loss
    2. Backward Pass:
        backward_and_get_grads -> output_layer_backward
        -> backward_transformer_block (N times reversed) -> backward_mlp
        -> backward_attention -> backward_layernorm -> softmax_rowwise_backward
    3. Parameters Update: adam_step
    """

    t = 1
    m, v = engine.adam_init(params)

    last_time = time.perf_counter()
    print("Training...")
    for step in range(1, epochs + 1):
        X_ids, Y_ids = utils.get_batch(encoded_data, block_size, batch_size)

        X = model.embed_tokens_and_positions(X_ids, params)
        loss, logits, probs, cache = model.forward(X, Y_ids, params, X_ids)
        grads = engine.backward_and_get_grads(cache)

        params, m, v, updates = engine.adam_step(
            params, grads, m, v, t, lr=learning_rate, beta1=0.9, beta2=0.999, eps=1e-8
        )

        t += 1

        if step == 1 or step % 100 == 0:
            now = time.perf_counter()
            elapsed = now - last_time
            print(f"step {step}, loss={loss:.6f} ({elapsed:.2f}s)")
            last_time = now

    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_WEIGHTS_PATH, "wb") as f:
        pickle.dump(params, f)
    print(f"Weights saved to {_WEIGHTS_PATH}")

    return params


def main():
    parser = argparse.ArgumentParser(description="BareGPT: NumPy-only GPT training")
    parser.add_argument(
        "--train", action="store_true", help="Force training even if weights exist"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training steps"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Tokens to generate"
    )
    parser.add_argument(
        "--data", type=str, default=_DATA_PATH, help="Path to input text file"
    )
    args = parser.parse_args()

    global epochs, learning_rate, batch_size
    if args.epochs is not None:
        epochs = args.epochs
    if args.lr is not None:
        learning_rate = args.lr
    if args.batch_size is not None:
        batch_size = args.batch_size

    # Initialization & Preparation
    text = prepare_input(args.data)
    encoded_data, vocab = tokenize_data(text)
    vocab_size = len(vocab)

    # Total parameters calculation
    n_params = (
        (vocab_size * d_model)
        + (block_size * d_model)
        + n_layers
        * (4 * d_model**2 + 4 * d_model + 2 * d_model * (4 * d_model) + 5 * d_model)
        + (d_model * vocab_size + vocab_size)
    )
    print("Model configuration:")
    print(f"- d_model:    {d_model:>10}")
    print(f"- n_layers:   {n_layers:>10}")
    print(f"- n_heads:    {n_heads:>10}")
    print(f"- batch_size: {batch_size:>10}")
    print(f"- block_size: {block_size:>10}")
    print(f"Total params: {n_params:>10,}\n")

    # Load trained weights if they exist (unless --train is specified)
    if not args.train:
        try:
            with open(_WEIGHTS_PATH, "rb") as f:
                params = pickle.load(f)
            print(f"Weights loaded from {_WEIGHTS_PATH}")
        except FileNotFoundError:
            params = model.init_weights(
                d_model, block_size, vocab_size, n_heads, n_layers
            )
            params = train(encoded_data, params, epochs, learning_rate)
    else:
        params = model.init_weights(d_model, block_size, vocab_size, n_heads, n_layers)
        params = train(encoded_data, params, epochs, learning_rate)

    # Inference & Generation
    n = len(encoded_data)
    L = 32
    start = np.random.randint(0, n - L + 1)
    sample = text[start : start + L]
    print("\nOutput:\n")

    viz_info = {}
    gen = model.generate_stream(
        sample, vocab, params, max_new_tokens=args.max_new_tokens, out_info=viz_info
    )
    for char in gen:
        print(char, end="", flush=True)

    print()  # Final newline

    # Attention visualization. Requires matplotlib
    # utils.plot_multi_head_attention(vocab, viz_info)


if __name__ == "__main__":
    main()
