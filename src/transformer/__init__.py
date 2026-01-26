"""
Encoder-Decoder Transformer implementation.

Based on "Attention is All You Need" (Vaswani et al., 2017)
and the Harvard NLP "Annotated Transformer" implementation.

This module provides the full encoder-decoder architecture for
sequence-to-sequence tasks like machine translation.

Key differences from GPT (decoder-only):
- Has both encoder and decoder
- Encoder uses bidirectional self-attention
- Decoder uses causal self-attention + cross-attention to encoder

Naming convention:
- Uses Annotated Transformer naming (d_model, h, N)
- nanoGPT equivalents: n_embd=d_model, n_head=h, n_layer=N
"""

from .attention import MultiHeadedAttention, attention, subsequent_mask
from .layers import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    SublayerConnection,
    LayerNorm,
    clones,
)
from .embeddings import Embeddings, PositionalEncoding
from .feedforward import PositionwiseFeedForward
from .generator import Generator
from .model import EncoderDecoder, make_model
from .decoding import greedy_decode, beam_search

__all__ = [
    # Main model
    "EncoderDecoder",
    "make_model",
    # Encoder/Decoder stacks
    "Encoder",
    "Decoder",
    "EncoderLayer",
    "DecoderLayer",
    # Attention
    "MultiHeadedAttention",
    "attention",
    "subsequent_mask",
    # Layers
    "SublayerConnection",
    "LayerNorm",
    "clones",
    # Embeddings
    "Embeddings",
    "PositionalEncoding",
    # Feed-forward
    "PositionwiseFeedForward",
    # Generator
    "Generator",
    # Decoding
    "greedy_decode",
    "beam_search",
]
