"""
Learning rate schedulers for transformer training.

Provides two scheduling strategies:
1. Cosine with warmup (nanoGPT style) - get_lr()
2. Noam scheduler (Annotated Transformer style) - rate()
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """
    nanoGPT-style learning rate scheduler with cosine decay.

    Args:
        it: Current iteration
        learning_rate: Base learning rate (n_embd equivalent: n_embd)
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of iterations to decay over
        min_lr: Minimum learning rate

    Returns:
        Learning rate for current iteration
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def rate(step, model_size, factor, warmup):
    """
    Noam-style learning rate scheduler from "Attention is All You Need".

    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    This increases linearly for the first warmup steps, and decreases
    thereafter proportionally to the inverse square root of step number.

    Args:
        step: Current training step (will default to 1 if 0 to avoid divide by zero)
        model_size: Model dimension (d_model / n_embd)
        factor: Scaling factor (typically 1.0)
        warmup: Number of warmup steps (typically 4000)

    Returns:
        Learning rate for current step

    Note:
        Naming aliases: model_size = d_model = n_embd
    """
    # Default step to 1 for LambdaLR function to avoid zero raising to negative power
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def get_std_opt(model, d_model=None, factor=1.0, warmup=4000):
    """
    Create the standard optimizer with Noam learning rate schedule.

    This is the optimizer configuration used in the original Transformer paper.

    Args:
        model: The model to optimize
        d_model: Model dimension. If None, tries to infer from model.
                 (Alias: n_embd)
        factor: Learning rate scaling factor
        warmup: Number of warmup steps

    Returns:
        tuple: (optimizer, lr_scheduler)
    """
    # Try to infer d_model from model if not provided
    if d_model is None:
        if hasattr(model, 'src_embed'):
            # EncoderDecoder model
            d_model = model.src_embed[0].d_model
        elif hasattr(model, 'config'):
            # GPT-style model
            d_model = model.config.n_embd
        else:
            raise ValueError("Could not infer d_model from model. Please provide it explicitly.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # Will be scaled by scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor, warmup)
    )

    return optimizer, lr_scheduler


# Naming compatibility aliases
noam_rate = rate  # Alias for clarity
cosine_lr = get_lr  # Alias for clarity
