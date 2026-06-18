---
title: Transformer Core
summary: Decoder-only transformer internals implemented in llm/transformer.py, including RMSNorm, RoPE, SwiGLU, custom attention, and custom loss.
slug: transformer-core
locale: en
group: core-stack
order: 3
translationKey: transformer-core
sourceFiles:
  - llm/transformer.py
sourceDocs:
  - docs/1.md
---

# Transformer Core

The entire base model stack lives in `llm/transformer.py`. The file is deliberately monolithic in the good sense: a reader can understand the core architecture, the loss, and the optimizer utilities without jumping across a large framework.

## File-Level Responsibilities

`llm/transformer.py` contains three layers of logic:

1. primitive trainable modules such as `Linear`, `Embedding`, and `RmsNorm`
2. compositional model blocks such as attention, RoPE, and `TransformerBlock`
3. optimization helpers such as `CrossEntropyLoss`, `AdamW`, `cos_lr_scheduler`, and `gradient_clip`

That design keeps the main training script extremely small because all model-local math is already owned here.

## Primitive Layers

### `Linear`

`Linear` stores a single weight matrix `self.w` with shape `[out_features, in_features]` and applies it with:

```python
einx.dot("... [in], out [in] -> ... out", x, self.w)
```

The initialization path uses truncated normal with:

```python
sigma = math.sqrt(2.0 / (in_features + out_features))
```

This is a straightforward affine projection without a bias term. The absence of bias simplifies the layer and matches common decoder-only implementations where bias-free projections are acceptable.

### `Embedding`

`Embedding` holds a parameter table `self.embeddings` of shape `[num_embeddings, embedding_dim]` and indexes it directly with integer token ids. Initialization uses truncated normal with standard deviation `1 / sqrt(embedding_dim)`.

There is no learned positional embedding table because the model relies on RoPE later in the stack.

### `RmsNorm`

`RmsNorm` is implemented explicitly rather than imported from a library:

1. cast input to `float32`
2. compute mean square along the last dimension
3. multiply by `rsqrt(variance + eps)`
4. scale by learnable gain `g`
5. cast back to the original dtype

Two details matter:

- RMSNorm normalizes by magnitude only; it does not subtract the mean.
- The float32 cast improves numerical stability for lower-precision training.

## Nonlinearities and Feed-Forward Path

### `SiLu`

`SiLu` is written directly as:

```python
torch.sigmoid(x) * x
```

### `SwiGlu`

`SwiGlu` is the feed-forward implementation actually used by the transformer. It defines:

- `w1`: gate input projection
- `w3`: value input projection
- `w2`: output projection back to model dimension

The forward rule is:

```python
self.w2(self.silu(self.w1(x)) * self.w3(x))
```

The file aliases:

```python
FFN = SwiGlu
```

So every transformer block uses SwiGLU as its MLP.

## Rotary Position Embeddings

`RoPE` precomputes cached cosine and sine tables at initialization time.

### Cache construction

The constructor builds:

- `inv_freq` for even dimensions
- `t = arange(max_seq_len)`
- `freqs = outer(t, inv_freq)`
- duplicated frequencies by `repeat_interleave`
- `cos_cached` and `sin_cached`

This turns positional encoding into a table lookup problem during forward.

### Forward rule

At runtime:

1. retrieve `cos` and `sin` rows for `token_positions`
2. reshape the last dimension into paired coordinates
3. rotate each pair `(a, b)` into `(-b, a)`
4. combine original and rotated versions via:

```python
x * cos + x_rotated * sin
```

If the input is 4-D, the code unsqueezes the trigonometric tables to align with the head dimension.

## Attention Stack

### `Softmax`

The file provides a custom stable softmax:

1. subtract row-wise max
2. exponentiate
3. divide by summed exponentials

This keeps the attention implementation self-contained.

### `ScaledDotProductAttention`

This module computes:

1. `att = QK^T`
2. `att_scale = att / sqrt(d_model)`
3. optional mask application with `masked_fill(mask, -1e9)`
4. softmax over the key dimension
5. weighted sum against `V`

The mask convention is explicit: `True` means "this position must not participate in softmax."

### `MultiHeadAttention`

This layer performs:

1. one projection from `d_model` to `3 * d_model`
2. split into `Q`, `K`, `V`
3. reshape into `[batch, heads, seq, head_dim]`
4. apply causal attention
5. reassemble heads
6. output projection back to `d_model`

The causal mask is precomputed once:

```python
torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
```

and sliced to `seq_len` at runtime.

### `MultiHeadAttentionWithRoPE`

This subclass reuses the same projection logic but inserts:

```python
q = self.rope(q, token_positions)
k = self.rope(k, token_positions)
```

before attention scores are computed.

If `token_positions` is omitted, the layer generates `0..seq_len-1` for each batch item. That makes the same attention module usable for both standard training and inference windows.

## Transformer Block Composition

Each `TransformerBlock` is pre-norm and residual:

1. `x_norm = rms_norm1(x)`
2. `x_atten = mult_head_atten(x_norm, token_positions)`
3. `x = x + x_atten`
4. `x_norm = rms_norm2(x)`
5. `x_ffe = ffe(x_norm)`
6. return `x + x_ffe`

This is a modern decoder-only structure:

- pre-norm for training stability
- RoPE in the attention path
- SwiGLU in the feed-forward path

## Full `Transformer`

The top-level model performs:

1. token-id lookup through `Embedding`
2. repeated application of `TransformerBlock`
3. final `RmsNorm`
4. projection to vocabulary logits with `out_linear`

If no positions are supplied, it builds a dense position tensor from sequence length.

The model therefore has a very small interface:

```python
forward(token_ids, token_positions=None) -> logits
```

That interface is reused by both `llm/training.py` and `llm/generating.py`.

## Loss Implementation

`CrossEntropyLoss` is also written out directly. It:

1. flattens logits to `[(...), vocab]`
2. flattens targets to `[(...)]`
3. computes `log_softmax`
4. indexes the correct-token log probability
5. averages the negative log likelihood

This keeps the training objective visible and avoids hiding a simple next-token loss behind a single library call.

## Optimizer and Scheduling Utilities

### `SGDDecay`

`SGDDecay` is a small educational optimizer where the effective step size decays like `lr / sqrt(t + 1)`.

### `AdamW`

The custom `AdamW` stores:

- `t`
- first moment `m`
- second moment `sm`

For each step it:

1. updates `m` and `sm`
2. computes bias-corrected `m_hat` and `sm_hat`
3. applies the Adam update
4. applies decoupled weight decay

That makes the optimizer state shape explicit and useful for understanding why optimizer memory becomes large in distributed training.

### `cos_lr_scheduler`

The scheduler has three regions:

- linear warmup up to `warmup_iters`
- cosine decay until `cos_cycle_iters`
- flat `lr_min` afterwards

### `gradient_clip`

`gradient_clip()` computes the global norm across all gradient tensors and scales them in-place if the norm exceeds `max_norm`.

## Architectural Character

The file's main value is not novelty. It is that modern decoder-only decisions are implemented in a single inspectable place:

- bias-free projections
- RMSNorm
- RoPE
- SwiGLU
- causal self-attention
- explicit next-token loss
- explicit optimizer state

That makes `llm/transformer.py` the best file in the repo for understanding how model internals and optimization math connect directly to the training loop.
