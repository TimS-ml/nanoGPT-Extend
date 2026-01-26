"""
Full Encoder-Decoder Transformer model.

Provides:
- EncoderDecoder: Main model class
- make_model: Factory function to create model from hyperparameters

Reference: "Attention is All You Need" (Vaswani et al., 2017)
"""

import copy
import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .layers import Encoder, Decoder, EncoderLayer, DecoderLayer
from .embeddings import Embeddings, PositionalEncoding
from .feedforward import PositionwiseFeedForward
from .generator import Generator


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.

    Base for sequence-to-sequence models like machine translation.

    Components:
        - encoder: Encoder stack
        - decoder: Decoder stack
        - src_embed: Source embedding + positional encoding
        - tgt_embed: Target embedding + positional encoding
        - generator: Final projection to vocabulary

    Args:
        encoder: Encoder module
        decoder: Decoder module
        src_embed: Source embedding module (nn.Sequential of Embeddings + PositionalEncoding)
        tgt_embed: Target embedding module (nn.Sequential of Embeddings + PositionalEncoding)
        generator: Generator module
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.

        Args:
            src: Source sequence, shape (batch, src_len)
            tgt: Target sequence, shape (batch, tgt_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)

        Returns:
            Decoder output, shape (batch, tgt_len, d_model)
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        Encode source sequence.

        Args:
            src: Source sequence, shape (batch, src_len)
            src_mask: Source attention mask

        Returns:
            Encoded representation, shape (batch, src_len, d_model)
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decode target sequence given encoded source.

        Args:
            memory: Encoded source, shape (batch, src_len, d_model)
            src_mask: Source attention mask
            tgt: Target sequence, shape (batch, tgt_len)
            tgt_mask: Target attention mask (causal)

        Returns:
            Decoded representation, shape (batch, tgt_len, d_model)
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(
    src_vocab,
    tgt_vocab,
    N=6,
    d_model=512,
    d_ff=2048,
    h=8,
    dropout=0.1,
):
    """
    Construct a model from hyperparameters.

    Creates a complete Encoder-Decoder Transformer with the specified
    configuration. Uses Xavier initialization for all parameters.

    Args:
        src_vocab: Source vocabulary size
        tgt_vocab: Target vocabulary size
        N: Number of encoder/decoder layers (nanoGPT: n_layer)
        d_model: Model dimension (nanoGPT: n_embd)
        d_ff: Feed-forward inner dimension (typically 4 * d_model)
        h: Number of attention heads (nanoGPT: n_head)
        dropout: Dropout probability

    Returns:
        EncoderDecoder model instance

    Note:
        Naming aliases:
        - N = n_layer
        - d_model = n_embd
        - h = n_head
        - d_ff = 4 * n_embd (default in nanoGPT)

    Example:
        >>> model = make_model(src_vocab=10000, tgt_vocab=10000)
        >>> # Or with nanoGPT-style naming:
        >>> model = make_model(
        ...     src_vocab=10000, tgt_vocab=10000,
        ...     N=6,        # n_layer
        ...     d_model=512,  # n_embd
        ...     h=8,        # n_head
        ... )
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg (Xavier uniform)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# Alias with nanoGPT-style parameter names
def make_transformer(
    src_vocab,
    tgt_vocab,
    n_layer=6,
    n_embd=512,
    n_head=8,
    dropout=0.1,
):
    """
    Construct a model using nanoGPT-style parameter names.

    This is an alias for make_model with nanoGPT naming convention.

    Args:
        src_vocab: Source vocabulary size
        tgt_vocab: Target vocabulary size
        n_layer: Number of encoder/decoder layers
        n_embd: Model dimension
        n_head: Number of attention heads
        dropout: Dropout probability

    Returns:
        EncoderDecoder model instance
    """
    return make_model(
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        N=n_layer,
        d_model=n_embd,
        d_ff=4 * n_embd,
        h=n_head,
        dropout=dropout,
    )
