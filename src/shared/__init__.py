"""
Shared components between GPT (decoder-only) and Transformer (encoder-decoder).
"""

from .lr_scheduler import get_lr, rate, get_std_opt
from .training import TrainState, Batch, run_epoch, DummyOptimizer, DummyScheduler
from .loss import LabelSmoothing, SimpleLossCompute

__all__ = [
    # Learning rate schedulers
    "get_lr",
    "rate",
    "get_std_opt",
    # Training utilities
    "TrainState",
    "Batch",
    "run_epoch",
    "DummyOptimizer",
    "DummyScheduler",
    # Loss functions
    "LabelSmoothing",
    "SimpleLossCompute",
]
