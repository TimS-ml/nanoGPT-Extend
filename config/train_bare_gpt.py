# BareGPT: NumPy-only GPT training on Shakespeare (character-level)
#
# This config documents the default hyperparameters for the BareGPT model.
# Unlike other configs in this repo, BareGPT does NOT use the configurator system
# (it's a standalone NumPy implementation with no PyTorch dependency).
#
# Usage (from repo root):
#   python -m src.bare_gpt.train              # train or load weights, then generate
#   python -m src.bare_gpt.train --train      # force re-training
#   python -m src.bare_gpt.train --epochs 500 # override training steps
#
# Default hyperparameters:
#   d_model = 256
#   n_layers = 2
#   n_heads = 4
#   batch_size = 16
#   block_size = 128
#   epochs = 700          # training steps (not full epochs over dataset)
#   learning_rate = 2.5e-4
#
# Architecture:
#   - Decoder-only, pre-norm Transformer (GPT-1 style)
#   - Character-level tokenization (no BPE)
#   - GELU activation (tanh approximation)
#   - No dropout, no weight decay, no LR scheduling
#   - Adam optimizer with optional gradient clipping
#   - ~1.64M parameters with default config
#
# Data:
#   Uses data/shakespeare_char/input.txt (same as train_shakespeare_char.py)
#
# Output:
#   Weights saved to out/bare_gpt/weights.pkl
