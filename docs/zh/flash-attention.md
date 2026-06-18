---
title: Flash Attention 与 Kernel 优化
summary: 解释 kernel/ 中的 Triton Flash Attention 路径、参考实现和性能对比脚本。
slug: flash-attention
locale: zh
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

# Flash Attention 与 Kernel 优化

`kernel/` 是仓库里的实验性性能层。这里的 Flash Attention 实现是真实可跑、可 benchmark 的，但它目前还没有接到默认的 `llm/transformer.py` 主路径里。所以这一章讲的是一个独立可复用 kernel，而不是“预训练默认已经启用的加速路径”。

## 职责边界

这一层被拆成三类文件：

- `kernel/flash_attention_triton.py`：Triton 前向 kernel 与 autograd 包装器
- `kernel/flash_attention_mock.py`：更易读的 PyTorch 参考实现
- `bench_mark/`：独立的性能测量脚本

这样优化实现、参考实现和 benchmark 可以分别演进，不会混成一个难维护的大文件。

## 为什么需要 Flash Attention

朴素 self-attention 往往会显式构造至少一个形状为 `[batch, query_len, key_len]` 的 score 矩阵，softmax 后还可能再产生一个同样大小的概率矩阵。因此：

- 临时显存开销随序列长度平方增长
- 带宽浪费在大矩阵读写上
- 长上下文下 attention 很快变成瓶颈

Flash Attention 通过改变执行顺序来解决这个问题：

1. 先把一个 query tile 留在片上
2. 逐块扫描 key/value tile
3. 维护 running softmax 统计量
4. 直接累计输出 tile
5. 避免显式保存完整 attention matrix

真正关键的不是“分块”本身，而是“分块时仍然保证 softmax 归一化正确”。

## 对外接口

Triton 路径通过一个 `torch.autograd.Function` 暴露：

```python
FlashAttention.apply(q, k, v, is_causal=False)
```

当前接口要求 `q`、`k`、`v` 的形状是 `[b, n, d]`。它不负责投影、不负责多头拆分、也不负责 RoPE。这些步骤必须由上游先完成。

## Triton 前向 Kernel

核心实现是 `flash_attention_forward_kernel`。launch grid 为：

```python
grid = (b, triton.cdiv(n, BQ))
```

这意味着每个 Triton program instance 负责：

- 一个 batch 元素，由 `pid_b` 指定
- 一个 query tile，由 `pid_tq` 指定

### block pointer 与内存布局

kernel 使用 `tl.make_block_ptr()` 明确描述 tile 的读写方式：

- `q_block_ptr`：读取 `[BQ, D]` 的 query tile
- `k_block_ptr`：把 key 看成 `[D, BK]`，这样 `tl.dot(q_i, k_j)` 直接得到 `[BQ, BK]`
- `v_block_ptr`：读取 `[BK, D]` 的 value tile
- `o_block_ptr`：写回 `[BQ, D]` 的输出 tile

这正是 Triton 存在的意义：代码显式控制 tile 形状、stride 和 pointer 的推进方式。

### running softmax 状态

在 key/value 循环开始前，kernel 会为每个 query row 初始化三类累加器：

```python
m_i = tl.full([BQ], value=float("-inf"), dtype=tl.float32)
l_i = tl.zeros([BQ], dtype=tl.float32)
o_i = tl.zeros([BQ, D], dtype=tl.float32)
```

它们的含义分别是：

- `m_i`：当前为止的行最大 attention score
- `l_i`：shift 后 softmax 分母的累计值
- `o_i`：当前输出累计值

在 tile 循环内，每个 key/value 块都会按标准 Flash Attention 递推更新：

```python
m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
scale = tl.exp(m_i - m_new)
p_ij = tl.exp(s_ij - m_new[:, None])

l_new = scale * l_i + tl.sum(p_ij, axis=1)
o_i = scale[:, None] * o_i + tl.dot(p_ij.to(v_j.dtype), v_j)
```

这就是整个文件最核心的数学部分。kernel 从来不需要把全局概率矩阵存下来，只保留足够重建最终归一化结果的统计量。

### causal mask

如果 `IS_CAUSAL` 为真，kernel 会同时在两个层面减少工作量。

第一，缩短 tile 循环终点：

```python
loop_end = tl.cdiv((pid_tq + 1) * BQ, BK)
```

这样当前 query tile 不会去扫描完全位于未来的 key tile。

第二，在当前可见 tile 内做局部 mask：

```python
offs_q = pid_tq * BQ + tl.arange(0, BQ)
offs_k = j * BK + tl.arange(0, BK)
s_ij += tl.where(offs_q[:, None] >= offs_k[None, :], 0, float("-inf"))
```

因此 causal 约束是在 tile 级别完成的，而不是先分配一整张全局 mask。

### 输出与保存状态

循环结束后，kernel 最终做：

```python
o_i /= l_i[:, None]
l_i = m_i + tl.log(l_i + eps)
```

其中：

- `o_i` 是最终输出 tile
- `l` buffer 保存的是每一行的 log-sum-exp 统计量

前向结束时，`q`、`k`、`v`、`l`、`o` 都会被存进 autograd context，供 backward 使用。

## Autograd 包装与反向传播

`FlashAttention` 是 `torch.autograd.Function` 的子类。前向走 Triton，但反向没有继续写 Triton kernel，而是调用：

```python
_flash_attn_backward_compiled(...)
```

这个 helper 用 `@torch.compile` 包装，内部仍然是 PyTorch 密集实现。它会：

1. 重建 `s = qk^T * scale`
2. 如果需要，应用 causal mask
3. 重建 `p = softmax(s)`
4. 计算 `dv`、`dp`、`ds`、`dq`、`dk`

所以当前优化边界是：

- 前向：Triton
- 反向：compiled PyTorch

这是个很实用的折中：前向演示了真正的 memory-saving kernel，而 backward 仍然保持可读和可调试。

## 参考实现：`flash_attention_mock.py`

`FlashAttentionMock` 是同一思想的可读版实现。它的前向流程是：

- 先用 `einx.rearrange("... n d -> (...) n d", q)` 展平 batch/head 前缀
- 在 Python 层双重循环遍历 query block 与 key block
- 用普通 PyTorch tensor 维护 `m_i`、`l_i`、`o_i`
- 最后再 reshape 回原始前缀形状

这个文件还提供了 `naive=True` 分支，直接做密集 attention。它的价值主要有三点：

- 在看 Triton 之前先理解 blockwise recurrence
- 做数值正确性对照
- 不依赖 Triton 也能跑通测试

## Benchmark 面

这一章配套两类 benchmark 入口。

### `bench_mark/bench_mark_flash_attention.py`

这个脚本会在不同组合上直接比较 Flash Attention kernel 与基线 attention：

- `dtype`
- `d_model`
- `seq_len`
- `batch_size`

它回答的是一个很窄但很重要的问题：在这些张量形状下，独立 kernel 比独立 baseline 快多少。

### `bench_mark/bench_mark_atten.py`

这个文件 benchmark 的是 `llm/transformer.py` 里的 `ScaledDotProductAttention`，并且可选开启 `torch.compile`。它提供了“普通 PyTorch attention”的基线。

## 和仓库其他部分的关系

目前 Flash Attention 还没有接到：

- `MultiHeadAttention`
- `MultiHeadAttentionWithRoPE`
- `llm/training.py`

默认模型路径仍然使用 `llm/transformer.py` 里那套更易读的 attention 实现。这样做的好处是：

- 端到端预训练路径仍然容易理解
- kernel 实验可以独立成熟

如果将来要集成，最自然的接点是在 `MultiHeadAttentionWithRoPE` 里，投影出 `q`、`k`、`v` 并应用 RoPE 之后。

## 当前实现的取舍

这条 kernel 路径目前有几项明确取舍：

- block size 固定为 `BQ=64`、`BK=64`
- 对外接口假设输入已经完成投影
- backward 用的是密集 PyTorch，而非 Triton
- 主模型还未集成这条 kernel

这些取舍和仓库目标是匹配的：重点不是交付一套完整生产 kernel 栈，而是用可运行代码把 Flash Attention 的数值技巧和内存布局讲清楚。
