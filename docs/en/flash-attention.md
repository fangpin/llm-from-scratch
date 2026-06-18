---
title: Flash Attention and Kernel Optimization
summary: How the Triton Flash Attention path in kernel/ reduces attention memory pressure and how it is validated against a reference implementation.
slug: flash-attention
locale: en
group: scale-performance
order: 5
translationKey: flash-attention
sourceFiles:
  - kernel/flash_attention_triton.py
  - kernel/flash_attention_mock.py
  - bench_mark/bench_mark_flash_attention.py
  - bench_mark/bench_mark_atten.py
sourceDocs:
  - docs/2.md
---

# Flash Attention and Kernel Optimization

`kernel/` is the repository's experimental performance layer. The Flash Attention implementation here is real and benchmarked, but it is not yet wired into the default `Transformer` path in `llm/transformer.py`. This chapter therefore explains a reusable kernel implementation, not a feature that pretraining already enables automatically.

## Ownership Boundary

The kernel work is split into three responsibilities:

- `kernel/flash_attention_triton.py` owns the Triton forward kernel and the autograd wrapper.
- `kernel/flash_attention_mock.py` owns a readable PyTorch blockwise reference path.
- `bench_mark/` owns the standalone performance measurements.

That separation is useful because optimized code, reference code, and benchmark code can evolve independently.

## Why Flash Attention Exists

Naive self-attention usually materializes at least one dense score matrix with shape `[batch, query_len, key_len]`, and often an equally large probability matrix after softmax. Memory traffic and temporary storage therefore scale quadratically with sequence length.

Flash Attention changes the execution order:

1. keep a query tile resident in on-chip memory
2. stream over key/value tiles
3. maintain running softmax statistics
4. accumulate the output tile directly
5. avoid storing the full attention matrix

The important detail is the running softmax state. Tiling alone is not enough; the algorithm also has to preserve the same normalization that dense softmax would have produced.

## Public Interface

The Triton path is exposed through a `torch.autograd.Function`:

```python
FlashAttention.apply(q, k, v, is_causal=False)
```

The current interface expects `q`, `k`, and `v` with shape `[b, n, d]`. Unlike `MultiHeadAttentionWithRoPE`, this function does not own projections, head reshaping, or RoPE. It assumes those steps have already happened upstream.

## Triton Forward Kernel

The core implementation is `flash_attention_forward_kernel`. The launch grid is:

```python
grid = (b, triton.cdiv(n, BQ))
```

Each Triton program instance therefore owns:

- one batch element, selected by `pid_b`
- one query tile, selected by `pid_tq`

### Block pointers and layout

The kernel uses `tl.make_block_ptr()` to define how each tile is read or written:

- `q_block_ptr` loads a `[BQ, D]` query tile
- `k_block_ptr` views keys as `[D, BK]` so `tl.dot(q_i, k_j)` yields `[BQ, BK]`
- `v_block_ptr` loads the matching `[BK, D]` value tile
- `o_block_ptr` writes the final `[BQ, D]` output tile

This is why the file lives in Triton instead of plain PyTorch. The implementation is explicitly controlling tile shapes, pointer strides, and tile advancement.

### Running softmax state

Before the key/value loop starts, the kernel initializes three accumulators per query row:

```python
m_i = tl.full([BQ], value=float("-inf"), dtype=tl.float32)
l_i = tl.zeros([BQ], dtype=tl.float32)
o_i = tl.zeros([BQ, D], dtype=tl.float32)
```

They mean:

- `m_i`: running row-wise max attention score
- `l_i`: running row-wise softmax denominator in shifted space
- `o_i`: running output accumulator

Inside the tile loop, each key/value block updates those states with the standard Flash Attention recurrence:

```python
m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
scale = tl.exp(m_i - m_new)
p_ij = tl.exp(s_ij - m_new[:, None])

l_new = scale * l_i + tl.sum(p_ij, axis=1)
o_i = scale[:, None] * o_i + tl.dot(p_ij.to(v_j.dtype), v_j)
```

This is the mathematical heart of the file. The kernel never materializes the global probability matrix; it only carries forward enough statistics to make the final normalization correct.

### Causal masking

If `IS_CAUSAL` is enabled, the kernel reduces work in two ways.

First, it shortens the tile loop:

```python
loop_end = tl.cdiv((pid_tq + 1) * BQ, BK)
```

So a query tile never scans key tiles that are entirely in the future.

Second, it masks the partially visible current tile:

```python
offs_q = pid_tq * BQ + tl.arange(0, BQ)
offs_k = j * BK + tl.arange(0, BK)
s_ij += tl.where(offs_q[:, None] >= offs_k[None, :], 0, float("-inf"))
```

Together those two branches implement the causal constraint without allocating a full mask tensor.

### Output and saved state

After the tile loop, the kernel finishes with:

```python
o_i /= l_i[:, None]
l_i = m_i + tl.log(l_i + eps)
```

`o_i` becomes the output tile. The `l` buffer stores the row-wise log-sum-exp term, which is saved for backward along with `q`, `k`, `v`, and `o`.

## Autograd Wrapper and Backward Path

`FlashAttention` subclasses `torch.autograd.Function`. The forward path launches Triton, but the backward path does not. Instead it calls:

```python
_flash_attn_backward_compiled(...)
```

which is decorated with `@torch.compile`.

That backward helper reconstructs dense attention in PyTorch:

1. rebuild `s = qk^T * scale`
2. apply the causal mask if needed
3. reconstruct `p = softmax(s)`
4. compute `dv`, `dp`, `ds`, `dq`, and `dk`

So the current optimization boundary is:

- Triton forward
- compiled PyTorch backward

This is a pragmatic compromise. The forward path shows the memory-saving algorithm, while backward stays easy to inspect and debug.

## Reference Implementation: `flash_attention_mock.py`

`FlashAttentionMock` is the readable version of the same blockwise idea. Its forward path:

- flattens batch/head prefixes with `einx.rearrange("... n d -> (...) n d", q)`
- iterates over query blocks and key blocks in Python
- maintains `m_i`, `l_i`, and `o_i` in ordinary PyTorch tensors
- reshapes the result back to the original prefix

The file also exposes a `naive=True` path that computes dense attention directly. That makes the mock implementation useful for:

- understanding the recurrence before reading Triton
- checking numerical agreement
- testing correctness without depending on the Triton kernel itself

## Benchmarking Surface

The kernel chapter is paired with two benchmark entrypoints.

### `bench_mark/bench_mark_flash_attention.py`

This script benchmarks the standalone Flash Attention kernel against the baseline attention path over combinations of:

- `dtype`
- `d_model`
- `seq_len`
- `batch_size`

It answers a narrow question: how fast is the isolated kernel compared with the isolated baseline under these tensor shapes?

### `bench_mark/bench_mark_atten.py`

This file benchmarks `ScaledDotProductAttention` from `llm/transformer.py`, optionally under `torch.compile`. It provides the "plain PyTorch" baseline that the Triton path is trying to beat.

## Relationship to the Rest of the Repository

At the moment, Flash Attention is not wired into:

- `MultiHeadAttention`
- `MultiHeadAttentionWithRoPE`
- `llm/training.py`

The default model path still uses the readable attention implementation from `llm/transformer.py`. That keeps the end-to-end pretraining stack easier to follow, and it lets the kernel work mature independently.

If you wanted to integrate the kernel, the bridge point would be inside `MultiHeadAttentionWithRoPE` after `q`, `k`, and `v` have been projected and RoPE has been applied.

## Practical Tradeoffs

The current kernel path makes a few explicit tradeoffs:

- tile sizes are fixed at `BQ=64` and `BK=64`
- the public interface expects already-projected tensors
- backward is dense PyTorch rather than Triton
- model integration is left to future work

That matches the repo's teaching goal. The point is not to provide a complete production kernel stack. The point is to show, in working code, how Flash Attention's numerical trick and memory layout actually work.
