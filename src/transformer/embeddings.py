"""
Token and positional embeddings.

Provides:
- Embeddings: Learned token embeddings with sqrt(d_model) scaling
- PositionalEncoding: Sinusoidal positional encodings

Reference: "Attention is All You Need" Section 3.4 and 3.5
"""

import math
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Learned token embeddings with sqrt(d_model) scaling.

    The scaling factor sqrt(d_model) is used to prevent the embeddings
    from becoming too small when d_model is large.

    Args:
        d_model: Model dimension (nanoGPT: n_embd)
        vocab: Vocabulary size (nanoGPT: vocab_size)
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Look up embeddings and scale by sqrt(d_model).

        Args:
            x: Token indices, shape (batch, seq_len)

        Returns:
            Embeddings, shape (batch, seq_len, d_model)
        """
        return self.lut(x) * math.sqrt(self.d_model)

    # Naming compatibility properties
    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.d_model

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self.lut.num_embeddings


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This allows the model to learn relative positions, as for any
    fixed offset k, PE(pos+k) can be represented as a linear function
    of PE(pos).

    Args:
        d_model: Model dimension (nanoGPT: n_embd)
        dropout: Dropout probability
        max_len: Maximum sequence length (nanoGPT: block_size)
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)

        Returns:
            Embeddings + positional encoding, same shape
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

    # Naming compatibility properties
    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.pe.size(-1)

    @property
    def d_model(self):
        """Model dimension."""
        return self.pe.size(-1)

    @property
    def block_size(self):
        """Maximum sequence length (nanoGPT naming)."""
        return self.pe.size(1)

    @property
    def max_len(self):
        """Maximum sequence length."""
        return self.pe.size(1)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding (alternative to sinusoidal).

    Similar to GPT-2's position embeddings.

    Args:
        d_model: Model dimension (nanoGPT: n_embd)
        dropout: Dropout probability
        max_len: Maximum sequence length (nanoGPT: block_size)
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Add learned positional encoding to input embeddings.

        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)

        Returns:
            Embeddings + positional encoding, same shape
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pe(positions)
        return self.dropout(x)

    # Naming compatibility properties
    @property
    def n_embd(self):
        """Alias for d_model (nanoGPT naming)."""
        return self.pe.embedding_dim

    @property
    def d_model(self):
        """Model dimension."""
        return self.pe.embedding_dim

    @property
    def block_size(self):
        """Maximum sequence length (nanoGPT naming)."""
        return self.pe.num_embeddings

    @property
    def max_len(self):
        """Maximum sequence length."""
        return self.pe.num_embeddings
