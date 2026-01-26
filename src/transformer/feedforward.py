"""
Position-wise Feed-Forward Network.

FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

Reference: "Attention is All You Need" Section 3.3
"""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Implements FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    The same feed-forward network is applied to each position independently.
    Uses ReLU activation (original paper) instead of GELU (GPT-2 style).

    Args:
        d_model: Model dimension (input and output) (nanoGPT: n_embd)
        d_ff: Inner layer dimension (typically 4 * d_model)
        dropout: Dropout probability (default: 0.1)

    Note:
        Naming aliases:
        - d_model = n_embd
        - d_ff = 4 * n_embd (in nanoGPT MLP)
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Output tensor, shape (..., d_model)
        """
        return self.w_2(self.dropout(self.w_1(x).relu()))

    # Naming compatibility properties
    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.w_1.in_features

    @property
    def d_model(self):
        """Model dimension."""
        return self.w_1.in_features

    @property
    def d_ff(self):
        """Inner layer dimension."""
        return self.w_1.out_features


class GELUFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation.

    Similar to GPT-2's MLP block.
    Uses GELU activation instead of ReLU.

    Args:
        d_model: Model dimension (input and output)
        d_ff: Inner layer dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(GELUFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Forward pass with GELU activation.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Output tensor, shape (..., d_model)
        """
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))

    # Naming compatibility properties
    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.w_1.in_features

    @property
    def d_model(self):
        """Model dimension."""
        return self.w_1.in_features

    @property
    def d_ff(self):
        """Inner layer dimension."""
        return self.w_1.out_features
