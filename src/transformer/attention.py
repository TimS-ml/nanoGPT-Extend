"""
Multi-Head Attention mechanism.

Implements:
- Scaled dot-product attention
- Multi-head attention
- Causal masking utilities

Reference: "Attention is All You Need" Section 3.2
"""

import math
import copy
import torch
import torch.nn as nn


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot-Product Attention'.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: Query tensor, shape (batch, heads, seq_len, d_k)
        key: Key tensor, shape (batch, heads, seq_len, d_k)
        value: Value tensor, shape (batch, heads, seq_len, d_v)
        mask: Optional mask tensor, shape (batch, 1, seq_len, seq_len) or broadcastable
        dropout: Optional dropout module

    Returns:
        tuple: (output, attention_weights)
            - output: shape (batch, heads, seq_len, d_v)
            - attention_weights: shape (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    """
    Create a mask to hide subsequent positions (for autoregressive decoding).

    This creates a lower triangular matrix where position i can only
    attend to positions <= i.

    Args:
        size: Sequence length

    Returns:
        Boolean mask tensor of shape (1, size, size)
        True values indicate positions that CAN be attended to.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def clones(module, N):
    """
    Produce N identical deep copies of a module.

    Args:
        module: PyTorch module to clone
        N: Number of copies

    Returns:
        nn.ModuleList containing N copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    Args:
        h: Number of attention heads (nanoGPT: n_head)
        d_model: Model dimension (nanoGPT: n_embd)
        dropout: Dropout probability (default: 0.1)

    Note:
        Naming aliases:
        - h = n_head
        - d_model = n_embd
        - d_k = d_model // h = head_size
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor, shape (batch, seq_len, d_model)
            key: Key tensor, shape (batch, seq_len, d_model)
            value: Value tensor, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor, shape (batch, seq_len, d_model)

        Note:
            For self-attention, query = key = value
            For cross-attention (decoder attending to encoder):
                query comes from decoder, key/value from encoder
        """
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch: d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value

        return self.linears[-1](x)

    # Naming compatibility properties
    @property
    def n_head(self):
        """Alias for h (nanoGPT naming)."""
        return self.h

    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.h * self.d_k

    @property
    def d_model(self):
        """Model dimension."""
        return self.h * self.d_k
