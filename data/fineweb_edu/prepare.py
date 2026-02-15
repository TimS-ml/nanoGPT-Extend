#!/usr/bin/env python3
"""
Download and tokenize FineWeb-Edu dataset sample for GPT training.

Downloads a small sample (~10B tokens worth) from HuggingFace,
tokenizes with GPT-2 BPE (tiktoken), and saves as uint16 .bin files.

Usage:
    python prepare.py                    # default: 10M token sample
    python prepare.py --num_tokens 1e9   # 1B tokens (larger)
    python prepare.py --num_tokens 1e7   # 10M tokens (quick test)

Output:
    train.bin  - training data (uint16)
    val.bin    - validation data (uint16)
"""

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_tokens",
        type=float,
        default=1e8,
        help="Approximate number of tokens to download (default 100M)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.01,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--subset", type=str, default="sample-10BT", help="Dataset subset/config"
    )
    args = parser.parse_args()

    num_tokens = int(args.num_tokens)
    print(f"Target: ~{num_tokens:,} tokens from {args.dataset}/{args.subset}")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    from datasets import load_dataset

    # Estimate: ~4 chars per token on average for English text
    # Each FineWeb doc has ~1-2K tokens, so we need roughly num_tokens / 500 documents
    # We'll stream and stop when we have enough tokens
    print("Loading dataset (streaming)...")
    ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)

    all_tokens = []
    total_tokens = 0
    doc_count = 0

    for doc in ds:
        text = doc["text"]
        tokens = enc.encode_ordinary(text)
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        doc_count += 1

        if doc_count % 10000 == 0:
            print(f"  Processed {doc_count:,} docs, {total_tokens:,} tokens...")

        if total_tokens >= num_tokens:
            break

    print(f"Total: {doc_count:,} documents, {total_tokens:,} tokens")

    # Convert to numpy array
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # Split into train/val
    val_size = int(len(all_tokens) * args.val_fraction)
    train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_tokens = all_tokens[-val_size:] if val_size > 0 else all_tokens[:1000]

    # Save
    train_path = os.path.join(os.path.dirname(__file__), "train.bin")
    val_path = os.path.join(os.path.dirname(__file__), "val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(
        f"Saved train.bin: {len(train_tokens):,} tokens ({os.path.getsize(train_path) / 1e6:.1f} MB)"
    )
    print(
        f"Saved val.bin:   {len(val_tokens):,} tokens ({os.path.getsize(val_path) / 1e6:.1f} MB)"
    )
    print("Done!")


if __name__ == "__main__":
    main()
