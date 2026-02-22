# BareGPT Core Functions Reference

Complete reference for every function in the `src/bare_gpt/` package.

---

## model.py -- Forward Pass & Generation

### `init_weights(d_model, block_size, vocab_size, n_heads, n_layers)`

**Line:** model.py:18

Initializes all trainable parameters for the transformer model.

**Parameters:**
- `d_model` (int): Embedding dimension (default 256)
- `block_size` (int): Maximum sequence length (default 128)
- `vocab_size` (int): Number of unique tokens in vocabulary
- `n_heads` (int): Number of attention heads (default 4)
- `n_layers` (int): Number of transformer blocks (default 2)

**Returns:** `params` (dict) -- flat dictionary containing all weights, biases,
and hyperparameters keyed by name.

**Weight layout per layer `i`:**

| Key              | Shape                  | Init Method    | Description               |
|------------------|------------------------|----------------|---------------------------|
| `W_q_{i}`        | (d_model, d_model)     | Xavier         | Query projection           |
| `W_k_{i}`        | (d_model, d_model)     | Xavier         | Key projection             |
| `W_v_{i}`        | (d_model, d_model)     | Xavier         | Value projection           |
| `W_o_{i}`        | (d_model, d_model)     | Xavier         | Output projection          |
| `W1_{i}`         | (d_model, 4*d_model)   | Xavier         | MLP expansion              |
| `b1_{i}`         | (1, 4*d_model)         | Zeros          | MLP expansion bias         |
| `W2_{i}`         | (4*d_model, d_model)   | Xavier         | MLP projection             |
| `b2_{i}`         | (1, d_model)           | Zeros          | MLP projection bias        |
| `ln1_gamma_{i}`  | (1, d_model)           | Ones           | LayerNorm scale (unused)   |
| `ln1_beta_{i}`   | (1, d_model)           | Zeros          | LayerNorm shift (unused)   |
| `ln2_gamma_{i}`  | (1, d_model)           | Ones           | LayerNorm scale (unused)   |
| `ln2_beta_{i}`   | (1, d_model)           | Zeros          | LayerNorm shift (unused)   |

**Global weights:**

| Key            | Shape                  | Init Method         | Description              |
|----------------|------------------------|----------------------|--------------------------|
| `W_out`        | (d_model, vocab_size)  | Xavier               | Output head projection   |
| `b_out`        | (vocab_size,)          | Zeros                | Output head bias         |
| `token_embed`  | (vocab_size, d_model)  | Normal (std=0.01)    | Token embedding table    |
| `pos_embed`    | (block_size, d_model)  | Normal (std=0.01)    | Position embedding table |

**Note:** LayerNorm gamma/beta are initialized but never applied in the forward
pass -- the forward LayerNorm uses bare normalization (effectively gamma=1,
beta=0). Gradients are still computed for them during backprop.

---

### `embed_tokens_and_positions(X_ids, params)`

**Line:** model.py:67

Computes combined token + positional embeddings via table lookup and addition.

**Parameters:**
- `X_ids` (ndarray): Token indices, shape `(B, T)`
- `params` (dict): Must contain `token_embed` and `pos_embed`

**Returns:** `X` (ndarray) -- shape `(B, T, d_model)`

**Logic:**
1. `tok_emb = token_embed[X_ids]` -- index into embedding table
2. `pos_emb = pos_embed[0:T]` -- slice positional embeddings for sequence length
3. Broadcast `pos_emb` across batch: `pos_emb[None, :, :]`
4. Return `tok_emb + pos_emb`

---

### `layernorm_forward(x, eps=1e-5)`

**Line:** model.py:101

Layer Normalization across the last dimension (features).

**Parameters:**
- `x` (ndarray): Input tensor, shape `(B, T, D)`
- `eps` (float): Small constant for numerical stability

**Returns:** `(x_norm, cache)`
- `x_norm` (ndarray): Normalized tensor, same shape as input
- `cache` (dict): Stores `x`, `mean`, `var`, `inv_std`, `eps` for backward pass

**Formula:** `x_norm = (x - mean) / sqrt(var + eps)`

---

### `multi_head_attention(X, params, layer_idx)`

**Line:** model.py:124

Computes masked multi-head self-attention for a single transformer layer.

**Parameters:**
- `X` (ndarray): Input hidden states, shape `(B, T, D)`
- `params` (dict): Model parameters (uses `W_q_{i}`, `W_k_{i}`, `W_v_{i}`, `W_o_{i}`)
- `layer_idx` (int): Which layer's weights to use

**Returns:** `(att_out, cache)`
- `att_out` (ndarray): Attention output, shape `(B, T, D)`
- `cache` (dict): All intermediates needed for backward (Q, K, V, heads, scores, probs, masks, weights)

**Steps:**
1. Linear projections: Q, K, V = X @ W_q, X @ W_k, X @ W_v
2. Reshape to multi-head: `(B, T, D)` -> `(B, H, T, Hd)` where `Hd = D // H`
3. Scaled dot-product scores: `(Qh @ Kh^T) / sqrt(Hd)`
4. Causal mask: upper-triangular mask sets future positions to `-1e10`
5. Stable softmax: subtract max, exponentiate, normalize
6. Weighted value aggregation: `att_probs @ Vh`
7. Concatenate heads: `(B, H, T, Hd)` -> `(B, T, D)`
8. Output projection: `att_out @ W_o`

---

### `mlp_forward(X, params, layer_idx)`

**Line:** model.py:218

Position-wise feed-forward network (two linear layers with GELU).

**Parameters:**
- `X` (ndarray): Input, shape `(B, T, D)`
- `params` (dict): Uses `W1_{i}`, `b1_{i}`, `W2_{i}`, `b2_{i}`
- `layer_idx` (int): Layer index

**Returns:** `(out, cache)`
- `out` (ndarray): MLP output, shape `(B, T, D)`
- `cache` (dict): Stores X, Z1, A1, Z2, weights for backward

**Steps:**
1. Expansion: `Z1 = X @ W1 + b1` -- `(B, T, D)` -> `(B, T, 4D)`
2. GELU (tanh approximation): `A1 = 0.5 * Z1 * (1 + tanh(sqrt(2/pi) * (Z1 + 0.044715 * Z1^3)))`
3. Projection: `Z2 = A1 @ W2 + b2` -- `(B, T, 4D)` -> `(B, T, D)`

---

### `transformer_block(X, params, layer_idx)`

**Line:** model.py:257

Executes one complete Pre-Norm transformer block.

**Parameters:**
- `X` (ndarray): Input hidden states, shape `(B, T, D)`
- `params` (dict): All model parameters
- `layer_idx` (int): Block index in the stack

**Returns:** `(Y, cache_block, att_matrix)`
- `Y` (ndarray): Block output, shape `(B, T, D)`
- `cache_block` (dict): Caches for both sub-layers (ln1, att, X2, ln2, mlp)
- `att_matrix` (ndarray): Attention output (for visualization)

**Flow:**
```
X -> LayerNorm -> Attention -> + (residual) -> X2
X2 -> LayerNorm -> MLP -> + (residual) -> Y
```

---

### `forward(X, targets, params, X_ids)`

**Line:** model.py:306

Full forward pass for training (with loss computation).

**Parameters:**
- `X` (ndarray): Embedded input, shape `(B, T, D)`
- `targets` (ndarray): Ground truth token IDs, shape `(B, T)`
- `params` (dict): All model parameters
- `X_ids` (ndarray): Original token IDs (stored in cache for embedding gradients)

**Returns:** `(loss, logits, probs, cache)`
- `loss` (float): Mean cross-entropy loss
- `logits` (ndarray): Raw scores, shape `(B, T, vocab_size)`
- `probs` (ndarray): Softmax probabilities, shape `(B, T, vocab_size)`
- `cache` (dict): Everything needed for `backward_and_get_grads`

**Flow:**
1. Pass through `n_layers` transformer blocks sequentially
2. Linear output head: `logits = Y @ W_out + b_out`
3. Stable softmax over vocabulary dimension
4. Cross-entropy loss: `-mean(log(probs[target_indices]))`

---

### `forward_no_loss(X, params)`

**Line:** model.py:382

Inference-only forward pass (no targets, no loss).

**Parameters:**
- `X` (ndarray): Embedded input, shape `(T, D)` or `(B, T, D)`
- `params` (dict): All model parameters

**Returns:** `(logits, probs, layer_caches, att_probs)`
- `logits` (ndarray): Shape `(T, vocab_size)` (squeezed)
- `probs` (ndarray): Shape `(T, vocab_size)` (squeezed)
- `layer_caches` (list): Per-layer caches
- `att_probs` (list): Attention weight matrices per layer for visualization

**Differences from `forward`:** Adds batch dim if missing, no loss computation,
returns attention matrices, squeezes batch dim on output.

---

### `generate_stream(prompt, vocab, params, max_new_tokens=512, k=20, temperature=0.8, out_info=None)`

**Line:** model.py:423

Generator that yields one character at a time using autoregressive decoding.

**Parameters:**
- `prompt` (str): Seed text
- `vocab` (list): Character vocabulary
- `params` (dict): Trained model parameters
- `max_new_tokens` (int): Maximum characters to generate
- `k` (int): Top-k sampling cutoff (None or 0 disables)
- `temperature` (float): Sampling temperature (1.0 = neutral)
- `out_info` (dict or None): If provided, populated with attention data for visualization

**Yields:** `str` -- one character per iteration

**See:** [inference_logic.md](inference_logic.md) for the complete step-by-step walkthrough.

---

## engine.py -- Backward Pass & Optimizer

### `adam_init(params)`

**Line:** engine.py:9

Initializes first-moment (`m`) and second-moment (`v`) accumulators for Adam,
each as zero-arrays matching the shape of every parameter.

**Parameters:**
- `params` (dict): Model parameters

**Returns:** `(m, v)` -- two dicts of zero arrays

---

### `adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=None)`

**Line:** engine.py:15

Performs one Adam optimizer update with optional global gradient clipping.

**Parameters:**
- `params` (dict): Current model parameters (modified in-place)
- `grads` (dict): Gradients for each parameter
- `m`, `v` (dict): Moment accumulators
- `t` (int): Current timestep (starting from 1, for bias correction)
- `lr` (float): Learning rate
- `beta1`, `beta2` (float): Exponential decay rates for moments
- `eps` (float): Numerical stability constant
- `clip_norm` (float or None): If set, clips global gradient norm to this value

**Returns:** `(params, m, v, updates)` -- updated parameters, moments, and the
last update dict (for debugging/logging)

**Steps:**
1. **Gradient clipping** (if `clip_norm` is set): compute global L2 norm, scale
   all gradients if it exceeds the threshold
2. **Per-parameter update**: for each parameter with a matching gradient:
   - Update biased first moment: `m = beta1 * m + (1-beta1) * g`
   - Update biased second moment: `v = beta2 * v + (1-beta2) * g^2`
   - Bias-correct: `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)`
   - Apply: `param -= lr * m_hat / (sqrt(v_hat) + eps)`

---

### `backward_and_get_grads(cache)`

**Line:** engine.py:76

Performs the complete backward pass through the entire model, returning
gradients for every trainable parameter.

**Parameters:**
- `cache` (dict): Global cache from `forward()`, containing layer caches,
  output head weights, targets, probs, and embedding tables

**Returns:** `grads` (dict) -- gradients keyed identically to `params`

**Steps:**
1. **Output head gradients**: `dlogits = (probs - one_hot) / (B*T)`, then
   compute `dW_out`, `db_out`, and `d_current` (gradient flowing back into the
   transformer stack)
2. **Transformer blocks in reverse**: iterate from layer `n_layers-1` down to 0,
   calling `backward_transformer_block` for each, accumulating per-layer weight
   gradients (`dW_q_{i}`, `dW_k_{i}`, etc.)
3. **Embedding gradients**: scatter `d_current` into `d_token_embed` using
   `np.add.at` (vectorized), sum across batch for `d_pos_embed`

---

### `backward_layernorm(dout, x, gamma=1, beta=1, eps=1e-5)`

**Line:** engine.py:179

Backward pass for LayerNorm. Recomputes forward statistics from `x` rather
than caching them (trades compute for memory).

**Parameters:**
- `dout` (ndarray): Upstream gradient, shape `(B, T, D)`
- `x` (ndarray): Original input to LayerNorm
- `gamma` (float/ndarray): Scale parameter (default 1, not used as trainable)
- `beta` (float/ndarray): Shift parameter (unused)
- `eps` (float): Stability constant

**Returns:** `(dx, dgamma, dbeta)`
- `dx` (ndarray): Gradient w.r.t. input, shape `(B, T, D)`
- `dgamma` (ndarray): Gradient w.r.t. scale, shape `(D,)`
- `dbeta` (ndarray): Gradient w.r.t. shift, shape `(D,)`

---

### `backward_transformer_block(dY, cache_block)`

**Line:** engine.py:213

Backward pass through one transformer block, reversing the forward computation
through MLP, LayerNorm, Attention, LayerNorm, and both residual connections.

**Parameters:**
- `dY` (ndarray): Upstream gradient, shape `(B, T, D)`
- `cache_block` (dict): Cached activations from the forward block

**Returns:** tuple of 11 elements:
```
(dX_block, dW_q, dW_k, dW_v, dW_o, dW1, db1, dW2, db2,
 (dgamma1, dbeta1), (dgamma2, dbeta2))
```

**Gradient flow:**
1. `dY` splits into residual path and MLP path
2. Backprop through MLP -> LayerNorm2 -> sum with residual = `dX2_total`
3. `dX2_total` splits into residual path and attention path
4. Backprop through Attention -> LayerNorm1 -> sum with residual = `dX_block`

---

### `backward_mlp(dY, cache)`

**Line:** engine.py:293

Backward pass through the two-layer MLP with GELU activation.

**Parameters:**
- `dY` (ndarray): Upstream gradient, shape `(B, T, D)`
- `cache` (dict): Forward pass intermediates (X, Z1, A1, W1, W2)

**Returns:** `(dX, dW1, db1, dW2, db2)`

**Steps:**
1. Backprop through second linear: `dW2 = A1^T @ dY`, `dA1 = dY @ W2^T`
2. Backprop through GELU: analytical derivative of tanh-approximated GELU
3. Backprop through first linear: `dW1 = X^T @ dZ1`, `dX = dZ1 @ W1^T`

Uses `np.einsum` for weight gradients to handle the batch dimension cleanly.

---

### `backward_attention(d_att_out, cache)`

**Line:** engine.py:324

Backward pass through multi-head masked self-attention. The most complex
gradient computation in the model.

**Parameters:**
- `d_att_out` (ndarray): Upstream gradient, shape `(B, T, D)`
- `cache` (dict): Forward intermediates (Qh, Kh, Vh, att_probs, weights, etc.)

**Returns:** `(dX, dW_q, dW_k, dW_v, dW_o)`

**Steps:**
1. Backprop through output projection W_o
2. Reshape to multi-head format `(B, H, T, Hd)`
3. Backprop through `att_probs @ Vh`: yields `d_att_probs` and `dVh`
4. Backprop through softmax: `d_scores = att_probs * (d_att_probs - sum(...))`
5. Backprop through scaling and QK^T dot product: yields `dQh`, `dKh`
6. Reshape back to `(B, T, D)` and compute weight gradients
7. Input gradient: `dX = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T`

---

## utils.py -- Tokenization, Batching, Initialization & Visualization

### `encode(text, vocab)`

**Line:** utils.py:4

Character-level encoder. Maps each character to its index in the vocabulary list.

**Parameters:**
- `text` (str): Input string
- `vocab` (list): Sorted list of unique characters

**Returns:** `ids` (list of int) -- token indices

---

### `decode(ids, vocab)`

**Line:** utils.py:14

Character-level decoder. Maps each index back to its character.

**Parameters:**
- `ids` (list of int): Token indices
- `vocab` (list): Same vocabulary used for encoding

**Returns:** `str` -- decoded string

---

### `get_batch(data, block_size, batch_size)`

**Line:** utils.py:25

Samples a random mini-batch of input-target pairs for training.

**Parameters:**
- `data` (list/ndarray): Encoded training data (integer token IDs)
- `block_size` (int): Sequence length per sample
- `batch_size` (int): Number of sequences per batch

**Returns:** `(x, y)` -- both shape `(batch_size, block_size)`
- `x`: Input sequences
- `y`: Target sequences (x shifted right by 1 position)

Random start indices are drawn uniformly from `[0, len(data) - block_size)`.

---

### `xavier_init(shape)`

**Line:** utils.py:42

Xavier/Glorot uniform initialization.

**Parameters:**
- `shape` (tuple): `(fan_in, fan_out)`

**Returns:** ndarray of shape `shape`, values in `[-limit, +limit]` where
`limit = sqrt(6 / (fan_in + fan_out))`

---

### `plot_multi_head_attention(vocab, viz_info)`

**Line:** utils.py:52

Generates a grid of heatmaps showing attention patterns for all heads in the
last transformer layer.

**Parameters:**
- `vocab` (list): Character vocabulary (for axis labels)
- `viz_info` (dict): Must contain `"ids"` (token IDs) and `"attentions"` (list
  of attention weight arrays per layer)

**Side effects:** Saves `multi_head_attention.png` to the current working directory.

**Note:** Imports `matplotlib.pyplot` lazily (inside the function) to avoid a
hard dependency.

---

## train.py -- Training Loop & CLI

### `prepare_input(path)`

**Line:** train.py:43

Reads a text file and strips whitespace.

**Parameters:**
- `path` (str): Path to input text file

**Returns:** `str` -- cleaned text content

---

### `tokenize_data(data)`

**Line:** train.py:51

Builds a character-level vocabulary from the data and encodes it.

**Parameters:**
- `data` (str): Raw text

**Returns:** `(ids, vocab)`
- `ids` (list of int): Encoded data
- `vocab` (list of str): Sorted unique characters

---

### `train(encoded_data, params, epochs, learning_rate)`

**Line:** train.py:57

Main training loop. Runs `epochs` steps of forward pass, backward pass, and
Adam update.

**Parameters:**
- `encoded_data` (list): Encoded training data
- `params` (dict): Model parameters (modified in-place)
- `epochs` (int): Number of training steps
- `learning_rate` (float): Adam learning rate

**Returns:** `params` (dict) -- updated parameters (also saved to
`out/bare_gpt/weights.pkl`)

**Per-step flow:**
1. `get_batch` -> random mini-batch
2. `embed_tokens_and_positions` -> input embeddings
3. `forward` -> loss, logits, probs, cache
4. `backward_and_get_grads` -> gradients
5. `adam_step` -> parameter update

Logs loss every 100 steps.

---

### `main()`

**Line:** train.py:103

CLI entry point. Parses arguments, loads or trains the model, and runs
generation.

**CLI arguments:**

| Flag               | Default                              | Description                          |
|--------------------|--------------------------------------|--------------------------------------|
| `--train`          | False                                | Force training even if weights exist |
| `--epochs`         | 700                                  | Number of training steps             |
| `--lr`             | 0.25e-3                              | Learning rate                        |
| `--batch_size`     | 16                                   | Batch size                           |
| `--max_new_tokens` | 1024                                 | Tokens to generate                   |
| `--data`           | `data/shakespeare_char/input.txt`    | Path to training data                |

**Flow:**
1. Parse args, override globals if flags provided
2. Load and tokenize data
3. Print model configuration and parameter count
4. If weights exist at `out/bare_gpt/weights.pkl` and `--train` not set: load them
5. Otherwise: initialize fresh weights and train
6. Sample a random 32-character seed from the training data
7. Stream-generate text using `generate_stream`
