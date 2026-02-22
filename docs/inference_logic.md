# BareGPT Inference Logic

How text generation works in BareGPT, step by step.

## Overview

BareGPT generates text autoregressively: it predicts one token at a time, appends
each prediction to its context, and repeats. The entire pipeline runs in NumPy
with no framework dependencies.

The entry point is `model.generate_stream()`, which yields one character at a time.

## Pipeline

```
prompt (string)
    |
    v
encode(prompt, vocab)          -- utils.py:4    character -> integer IDs
    |
    v
[generation loop: max_new_tokens iterations]
    |
    +-- crop to last block_size tokens
    |
    +-- token_embed[context_ids] + pos_embed[0:T]    -- embedding lookup
    |
    +-- forward_no_loss(X, params)                    -- model.py:382
    |       |
    |       +-- [for each layer 0..n_layers-1]
    |       |       |
    |       |       +-- layernorm_forward(X)          -- model.py:101
    |       |       +-- multi_head_attention(X_norm)   -- model.py:124
    |       |       +-- residual: X2 = X + att_out
    |       |       +-- layernorm_forward(X2)
    |       |       +-- mlp_forward(X2_norm)           -- model.py:218
    |       |       +-- residual: Y = X2 + mlp_out
    |       |
    |       +-- logits = Y @ W_out + b_out             -- linear projection
    |       +-- probs  = softmax(logits)               -- stable softmax
    |
    +-- p = probs[-1]          -- take last token's distribution
    |
    +-- temperature scaling    -- adjusts sharpness of distribution
    |
    +-- top-k sampling         -- restrict to k most probable tokens
    |
    +-- np.random.choice       -- sample next token ID
    |
    +-- decode([next_id])      -- integer ID -> character
    |
    +-- yield character        -- stream to caller
    |
    +-- append next_id to context
    |
    v
[loop end]
```

## Detailed Walkthrough

### 1. Prompt Encoding

```python
ids = utils.encode(prompt, vocab)    # "hello" -> [46, 43, 50, 50, 53]
```

Each character is mapped to its index in the sorted vocabulary list (all unique
characters in the training corpus). There is no BPE or subword tokenization --
BareGPT operates at the character level.

### 2. Context Windowing

```python
context_ids = ids[-block_size:]
```

The model can only attend to `block_size` tokens (default 128) due to the
learned positional embeddings. If the accumulated sequence is longer, only
the most recent `block_size` tokens are kept.

### 3. Embedding Construction

```python
tok_emb = token_embed[context_ids]      # (T, d_model)
pos_emb = pos_embed[np.arange(T)]       # (T, d_model)
X = tok_emb + pos_emb                   # (T, d_model)
```

Token embeddings capture "what" each token is. Positional embeddings capture
"where" it sits in the sequence. Their element-wise sum produces the input
representation for the transformer stack.

### 4. Forward Pass (`forward_no_loss`)

This is the inference-specific forward pass (no loss computation, no target
tokens needed). It adds a batch dimension if missing (`X.ndim == 2 -> X[None]`),
runs through all transformer blocks, and projects to vocabulary logits.

**Key difference from the training forward pass (`forward`):**
- No cross-entropy loss calculation
- No target token indexing
- Returns attention probabilities for optional visualization
- Squeezes the batch dimension on output (returns shapes `(T, vocab)`)

#### 4a. Transformer Block (repeated `n_layers` times)

Each block follows the Pre-Norm architecture:

```
X ----+---> LayerNorm ---> Multi-Head Attention ---> + ---> X2
      |                                              |
      +----------------------------------------------+  (residual)

X2 ---+---> LayerNorm ---> MLP (FFN) ---> + ---> Y
      |                                    |
      +------------------------------------+  (residual)
```

**LayerNorm** (`layernorm_forward`):
Normalizes across the feature dimension (d_model). Stabilizes activations
and helps gradient flow. Note: gamma/beta affine parameters are initialized
but the forward pass applies bare normalization only (gamma=1, beta=0 in effect).

**Multi-Head Attention** (`multi_head_attention`):
1. Linear projections: Q = X @ W_q, K = X @ W_k, V = X @ W_v
2. Reshape to (B, H, T, Hd) for parallel head computation
3. Scaled dot-product: scores = (Q @ K^T) / sqrt(Hd)
4. Causal masking: future positions set to -1e10 (becomes ~0 after softmax)
5. Softmax: normalize scores to attention weights (probabilities)
6. Weighted sum: att_out = att_weights @ V
7. Concatenate heads and project: output = concat @ W_o

**MLP** (`mlp_forward`):
1. Expand: Z1 = X @ W1 + b1 (d_model -> 4*d_model)
2. GELU activation (tanh approximation)
3. Project: Z2 = A1 @ W2 + b2 (4*d_model -> d_model)

#### 4b. Output Head

```python
logits = Y @ W_out + b_out    # (1, T, vocab_size)
```

Projects the final hidden state to a score for every token in the vocabulary.

#### 4c. Softmax

```python
logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
exp_logits = np.exp(logits_shifted)
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
```

Numerically stable softmax: subtract the max before exponentiating to prevent
overflow.

### 5. Sampling

Only the last token's probability distribution matters (the model's prediction
for what comes next):

```python
p = probs[-1].astype(np.float64)
```

**Temperature** (default 0.8): Sharpens (< 1) or flattens (> 1) the distribution.
Applied in log-space for numerical stability:

```python
p = np.exp(np.log(p + 1e-10) / temperature)
p /= p.sum()
```

**Top-k** (default k=20): Restricts sampling to the k most probable tokens.
Uses `np.argpartition` for O(n) selection (not full sort):

```python
top_idx = np.argpartition(p, -k)[-k:]
top_probs = p[top_idx]
top_probs /= top_probs.sum()
next_id = np.random.choice(top_idx, p=top_probs)
```

### 6. Streaming Output

The newly sampled token ID is:
1. Appended to the running context (`ids.append(int(next_id))`)
2. Decoded to a character (`utils.decode([next_id], vocab)`)
3. Yielded to the caller (enabling real-time streaming)

The loop repeats, now with one more token in context, until `max_new_tokens`
tokens have been generated.

### 7. Attention Visualization (Optional)

If an `out_info` dict is passed, `generate_stream` populates it with the
attention weight matrices from the last generation step. This can be fed to
`utils.plot_multi_head_attention()` to visualize which tokens the model
attended to.

## Default Configuration

| Parameter        | Value   | Description                              |
|------------------|---------|------------------------------------------|
| `d_model`        | 256     | Embedding / hidden dimension             |
| `n_layers`       | 2       | Number of transformer blocks             |
| `n_heads`        | 4       | Number of attention heads                |
| `block_size`     | 128     | Maximum context length                   |
| `vocab_size`     | ~65     | Unique characters in Shakespeare corpus  |
| `temperature`    | 0.8     | Sampling temperature                     |
| `k`              | 20      | Top-k sampling cutoff                    |
| `max_new_tokens` | 512/1024| Tokens to generate (stream default / CLI)|

## Data Flow Shapes (Single Sequence, B=1)

```
Input IDs:     (T,)           e.g. (128,)
Token embed:   (T, 256)
Pos embed:     (T, 256)
X (combined):  (1, T, 256)    batch dim added by forward_no_loss
Per-layer:     (1, T, 256)    maintained through residual connections
Att scores:    (1, 4, T, T)   per-head attention matrix
MLP hidden:    (1, T, 1024)   4x expansion
Logits:        (T, 65)        squeezed on return
Probs:         (T, 65)        squeezed on return
Output:        scalar (int)   sampled token ID
```
