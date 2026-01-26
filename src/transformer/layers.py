"""
Encoder and Decoder layer implementations.

Provides:
- LayerNorm: Layer normalization
- SublayerConnection: Residual connection + layer norm
- EncoderLayer: Self-attention + feed-forward
- DecoderLayer: Self-attention + cross-attention + feed-forward
- Encoder: Stack of encoder layers
- Decoder: Stack of decoder layers

Reference: "Attention is All You Need" Section 3.1
"""

import copy
import torch
import torch.nn as nn


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


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Normalizes the input tensor along the last dimension.
    Unlike PyTorch's LayerNorm, this uses explicit parameters a_2 and b_2.

    Args:
        features: Number of features (d_model)
        eps: Epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.

    Implements: x + Dropout(Sublayer(LayerNorm(x)))

    Note: For code simplicity, the norm is applied first (Pre-LN)
    as opposed to last (Post-LN) as in the original paper.

    Args:
        size: Model dimension (d_model)
        dropout: Dropout probability
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.

        Args:
            x: Input tensor
            sublayer: A callable (lambda or function) that takes normalized x

        Returns:
            x + Dropout(sublayer(Norm(x)))
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward.

    Architecture:
        1. Multi-head self-attention
        2. Position-wise feed-forward network

    Both sublayers use residual connections and layer normalization.

    Args:
        size: Model dimension (d_model)
        self_attn: Multi-head attention module
        feed_forward: Position-wise feed-forward module
        dropout: Dropout probability
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Source attention mask

        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and feed-forward.

    Architecture:
        1. Masked multi-head self-attention (causal)
        2. Multi-head cross-attention (attending to encoder output)
        3. Position-wise feed-forward network

    All sublayers use residual connections and layer normalization.

    Args:
        size: Model dimension (d_model)
        self_attn: Multi-head self-attention module
        src_attn: Multi-head cross-attention module
        feed_forward: Position-wise feed-forward module
        dropout: Dropout probability
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Forward pass through decoder layer.

        Args:
            x: Input tensor (decoder), shape (batch, tgt_seq_len, d_model)
            memory: Encoder output, shape (batch, src_seq_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)

        Returns:
            Output tensor, shape (batch, tgt_seq_len, d_model)
        """
        m = memory
        # Self-attention (causal) on decoder input
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Cross-attention to encoder output
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # Feed-forward
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N identical layers.

    Args:
        layer: EncoderLayer instance to clone
        N: Number of layers (nanoGPT: n_layer)
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Encoded output, shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.

    Args:
        layer: DecoderLayer instance to clone
        N: Number of layers (nanoGPT: n_layer)
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass through each decoder layer.

        Args:
            x: Target input tensor, shape (batch, tgt_seq_len, d_model)
            memory: Encoder output, shape (batch, src_seq_len, d_model)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)

        Returns:
            Decoded output, shape (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
