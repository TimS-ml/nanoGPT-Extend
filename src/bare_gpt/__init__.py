"""
BareGPT: A minimalist, NumPy-only implementation of a GPT-style Transformer.

Inspired by NanoGPT (Karpathy, 2023) and GPT-1 (Radford et al., 2018) for the
decoder-only generative approach, and Attention Is All You Need
(Vaswani et al., 2017) for the core Transformer mechanics.

Original implementation by Damien Boureille, 2025 (MIT Licence).
Integrated into nanoGPT-Extend as a subpackage.

Usage:
    python -m src.bare_gpt.train          # Train from scratch or generate from saved weights
    python -m src.bare_gpt.train --train  # Force training even if weights exist
"""
