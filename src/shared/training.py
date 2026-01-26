"""
Training utilities for transformer models.

Provides:
- TrainState: Track training progress
- Batch: Wrapper for batched data with masks
- run_epoch: Generic training loop for encoder-decoder models
- Dummy optimizer/scheduler for evaluation
"""

import time
import torch


class TrainState:
    """Track number of steps, examples, and tokens processed."""

    def __init__(self):
        self.step: int = 0          # Steps in the current epoch
        self.accum_step: int = 0    # Number of gradient accumulation steps
        self.samples: int = 0       # Total number of examples used
        self.tokens: int = 0        # Total number of tokens processed


class Batch:
    """
    Object for holding a batch of data with mask during training.

    For encoder-decoder models, handles source and target sequences
    with appropriate masking.

    Args:
        src: Source sequence tensor (batch, seq_len)
        tgt: Target sequence tensor (batch, seq_len), optional
        pad: Padding token index (default: 2 = <blank>)
    """

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]      # Input to decoder (shifted right)
            self.tgt_y = tgt[:, 1:]      # Target for loss (original shifted left)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
        else:
            self.tgt = None
            self.tgt_y = None
            self.tgt_mask = None
            self.ntokens = 0

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        from src.transformer.attention import subsequent_mask

        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class DummyOptimizer(torch.optim.Optimizer):
    """Dummy optimizer for evaluation mode."""

    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    """Dummy scheduler for evaluation mode."""

    def step(self):
        pass


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=None,
):
    """
    Train a single epoch for encoder-decoder models.

    Args:
        data_iter: Iterator yielding Batch objects
        model: The model to train (EncoderDecoder)
        loss_compute: Loss computation function (e.g., SimpleLossCompute)
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        mode: "train", "train+log", or "eval"
        accum_iter: Gradient accumulation steps
        train_state: TrainState object for tracking progress

    Returns:
        tuple: (average_loss, train_state)
    """
    if train_state is None:
        train_state = TrainState()

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | "
                f"Loss: {loss / batch.ntokens:6.2f} | "
                f"Tokens / Sec: {tokens / elapsed:7.1f} | "
                f"Learning Rate: {lr:6.1e}"
            )
            start = time.time()
            tokens = 0

        del loss
        del loss_node

    return total_loss / total_tokens, train_state


def data_gen(V, batch_size, nbatches, device=None):
    """
    Generate random data for a src-tgt copy task.

    Useful for testing and debugging the model.

    Args:
        V: Vocabulary size
        batch_size: Batch size
        nbatches: Number of batches to generate
        device: Device to place tensors on

    Yields:
        Batch objects for training
    """
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1  # Start token
        src = data.clone().detach()
        tgt = data.clone().detach()

        if device is not None:
            src = src.to(device)
            tgt = tgt.to(device)

        yield Batch(src, tgt, pad=0)
