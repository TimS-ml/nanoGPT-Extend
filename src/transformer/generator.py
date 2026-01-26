"""
Generator module for producing output probabilities.

Implements the final linear projection + log softmax for generating
token probabilities.
"""

import torch.nn as nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """
    Define standard linear + log_softmax generation step.

    Projects from d_model to vocabulary size and applies log_softmax.

    Args:
        d_model: Model dimension (nanoGPT: n_embd)
        vocab: Vocabulary size (nanoGPT: vocab_size)
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        Project to vocabulary and apply log_softmax.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Log probabilities, shape (..., vocab_size)
        """
        return log_softmax(self.proj(x), dim=-1)

    # Naming compatibility properties
    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.proj.in_features

    @property
    def d_model(self):
        """Model dimension."""
        return self.proj.in_features

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self.proj.out_features
