"""
Loss functions for transformer training.

Provides:
- LabelSmoothing: KL-divergence loss with label smoothing
- SimpleLossCompute: Simple loss computation wrapper
"""

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing using KL divergence loss.

    Instead of using a one-hot target distribution, creates a distribution
    that has `confidence` probability on the correct word and the rest of
    the smoothing mass distributed throughout the vocabulary.

    Label smoothing hurts perplexity (model becomes less confident) but
    improves accuracy and BLEU score.

    Args:
        size: Vocabulary size
        padding_idx: Index of padding token (will be zeroed in target)
        smoothing: Smoothing factor (default: 0.0 = no smoothing)
                   Typical value: 0.1
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        Compute smoothed label loss.

        Args:
            x: Model predictions (log probabilities), shape (batch * seq, vocab)
            target: Target indices, shape (batch * seq,)

        Returns:
            KL divergence loss
        """
        assert x.size(1) == self.size

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """
    A simple loss compute and train function.

    Wraps the generator and criterion for easy loss computation.

    Args:
        generator: Generator module (linear + log_softmax)
        criterion: Loss criterion (e.g., LabelSmoothing)
    """

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        """
        Compute loss for a batch.

        Args:
            x: Model output (before generator), shape (batch, seq, d_model)
            y: Target indices, shape (batch, seq)
            norm: Normalization factor (typically number of non-pad tokens)

        Returns:
            tuple: (loss_data * norm, loss_node for backprop)
        """
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)),
                y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


class CrossEntropyLossCompute:
    """
    Standard cross-entropy loss computation (no label smoothing).

    Compatible with SimpleLossCompute interface but uses standard
    cross-entropy instead of KL divergence with smoothing.

    Args:
        generator: Generator module (linear + log_softmax)
        padding_idx: Index of padding token to ignore
    """

    def __init__(self, generator, padding_idx=0):
        self.generator = generator
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            reduction="sum"
        )

    def __call__(self, x, y, norm):
        """
        Compute cross-entropy loss for a batch.

        Args:
            x: Model output (before generator), shape (batch, seq, d_model)
            y: Target indices, shape (batch, seq)
            norm: Normalization factor

        Returns:
            tuple: (loss_data * norm, loss_node for backprop)
        """
        # Generator gives log_softmax, but CrossEntropyLoss expects logits
        # So we use the projection directly
        logits = self.generator.proj(x)
        loss = (
            self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                y.contiguous().view(-1)
            )
            / norm
        )
        return loss.data * norm, loss
