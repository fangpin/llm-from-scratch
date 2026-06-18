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

The core model lives in `llm/transformer.py`. The file is not only a `Transformer` class. It also defines the primitive layers and utility logic that the training script depends on:

- custom `Linear`
- `Embedding`
- `RmsNorm`
- `Softmax`
- attention modules
- `RoPE`
- `SwiGlu`
- `CrossEntropyLoss`
- custom optimizer utilities

## Custom Linear and Embedding Layers

`Linear` stores a single learnable weight matrix and applies it with:

```python
einx.dot("... [in], out [in] -> ... out", x, self.w)
```

The implementation uses truncated normal initialization when pretrained weights are not supplied.

`Embedding` keeps a single learnable embedding table and indexes it directly with token ids.

These layers are simple by design. The point is not to out-feature `torch.nn`. The point is to make the data flow visible.

## RMSNorm Instead of LayerNorm

`RmsNorm` casts the input to float32, computes the mean square on the last dimension, rescales the activation, and applies a learned gain vector `g`.

Two details matter here:

1. it does not subtract the mean, unlike LayerNorm
2. it restores the original input dtype at the end

That matches the repo's "modern decoder-only baseline" goal instead of a historical Transformer baseline.

## Attention Stack

### `ScaledDotProductAttention`

The attention module:

1. computes `QK^T`
2. divides by `sqrt(d_model)`
3. applies an optional mask with `masked_fill(mask, -1e9)`
4. runs a custom softmax
5. multiplies attention scores by `V`

The mask convention is explicit: `true` means the position should be masked out.

### `MultiHeadAttention`

This layer projects the input once into concatenated `QKV`, then reshapes with `einx.rearrange` into head-major tensors.

It also precomputes a causal upper-triangular mask and slices it to the active sequence length:

```python
causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1)
```

That keeps the implementation decoder-only and autoregressive.

### `MultiHeadAttentionWithRoPE`

This subclass reuses the base multi-head logic and inserts RoPE on `q` and `k` before the attention kernel runs.

If no `token_positions` are passed, it creates a default `0..seq_len-1` position tensor for each batch item.

## Rotary Position Embeddings

`RoPE` precomputes cached cosine and sine tables up to `max_seq_len`. During forward:

1. it looks up the cached position-specific cos and sin values
2. it reshapes the head dimension into paired coordinates
3. it rotates each pair `(a, b)` into `(-b, a)`
4. it combines original and rotated vectors with the cached trig values

That means positional information is injected by rotating query and key vectors rather than by adding learned position embeddings to token embeddings.

## Feed-Forward Network

The file aliases:

```python
FFN = SwiGlu
```

`SwiGlu` uses three linear layers:

- `w1`
- `w3`
- `w2`

The forward path is:

```python
self.w2(self.silu(self.w1(x)) * self.w3(x))
```

That is the repo's feed-forward block and matches the stated use of SwiGLU rather than ReLU or GELU.

## Transformer Block Composition

Each `TransformerBlock` is pre-norm:

1. normalize
2. attention
3. residual add
4. normalize again
5. feed-forward
6. residual add

That structure is visible in the block implementation and aligns with the README claim that the project uses a modern decoder-only layout.

## Transformer Model

`Transformer` itself is straightforward:

1. token ids -> embeddings
2. run all blocks
3. final RMSNorm
4. project back to vocabulary logits

If `token_positions` are not supplied, the model generates them from sequence length.

This keeps generation and training code paths aligned: both can call the same model interface and optionally override token positions when needed.

## Custom Loss and Optimizer Utilities

`CrossEntropyLoss` explicitly reshapes logits and targets, computes `log_softmax`, indexes the correct-token log probability, and averages the negative log likelihood.

The file also defines:

- `SGDDecay`
- `AdamW`
- `cos_lr_scheduler`
- `gradient_clip`

That is important because the repo's "from scratch" scope includes not just the forward model, but also the optimization tools around it.
