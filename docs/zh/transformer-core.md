---
title: Transformer 核心
summary: 解释 llm/transformer.py 中的 decoder-only 结构，包括 RMSNorm、RoPE、SwiGLU、自定义 attention 和自定义 loss。
slug: transformer-core
locale: zh
group: core-stack
order: 3
translationKey: transformer-core
sourceFiles:
  - llm/transformer.py
sourceDocs:
  - docs/1.md
---

# Transformer 核心

核心模型实现集中在 `llm/transformer.py`。这个文件不只是一个 `Transformer` 类，而是把构成 decoder-only 模型的主要积木都写在一起：

- `Linear`
- `Embedding`
- `RmsNorm`
- `Softmax`
- attention 模块
- `RoPE`
- `SwiGlu`
- `CrossEntropyLoss`
- 自定义优化器工具

## 基础层

### `Linear`

`Linear` 只维护一个权重矩阵，用 `einx.dot` 做矩阵乘。

如果没有传入现成权重，就会使用截断正态分布初始化。这个实现的重点不是功能多，而是把线性层真正需要做的事情暴露出来。

### `Embedding`

`Embedding` 维护一张可学习的 embedding table，前向时直接用 token ids 做索引。

这也让模型的输入路径非常直白：

`token_ids -> embeddings`

## RMSNorm

`RmsNorm` 的前向逻辑是：

1. 先把输入转成 float32
2. 计算最后一维的均方
3. 用 `rsqrt` 做缩放
4. 乘以可学习参数 `g`
5. 再转回原始 dtype

它和 LayerNorm 的差异在于不减均值，只做均方归一化。这正是这个仓库选择更现代 decoder-only 配置的一个体现。

## Attention 路径

### `ScaledDotProductAttention`

注意力的核心过程完全展开在代码里：

1. 计算 `QK^T`
2. 除以 `sqrt(d_model)`
3. 如果有 mask，就把被遮掉的位置填成 `-1e9`
4. 经过自定义 softmax
5. 再乘上 `V`

这里的 mask 约定也很清楚：`true` 表示该位置不参与 softmax。

### `MultiHeadAttention`

这个模块会先一次性投影出拼在一起的 `QKV`，再用 `einx.rearrange` 拆成多头结构。

同时，它在初始化时就缓存了 causal mask，并在前向时按当前序列长度裁切，所以模型天然是 autoregressive decoder-only。

### `MultiHeadAttentionWithRoPE`

这个类在多头注意力基础上加入 RoPE，对 `q` 和 `k` 做旋转位置编码，然后再进入注意力计算。

如果外部没有传 `token_positions`，它会自动构造 `0..seq_len-1` 的位置张量。

## RoPE

`RoPE` 在初始化时会预先缓存所有位置的 cosine 和 sine 表。

前向时的关键步骤是：

1. 按 token position 取出对应 cos/sin
2. 把最后一维拆成成对坐标
3. 把 `(a, b)` 旋转成 `(-b, a)`
4. 用 trig 缓存组合原向量和旋转向量

所以这里的位置编码不是把位置 embedding 加到 token embedding 上，而是直接作用在注意力里的 `q` / `k` 向量。

## SwiGLU 前馈层

文件里直接定义了：

```python
FFN = SwiGlu
```

`SwiGlu` 使用三层线性变换：

- `w1`
- `w3`
- `w2`

其核心前向为：

```python
self.w2(self.silu(self.w1(x)) * self.w3(x))
```

这也是 README 里强调的现代 FFN 结构来源。

## Block 结构

`TransformerBlock` 是典型 pre-norm 结构：

1. 先 norm
2. 进入 attention
3. 做 residual add
4. 再 norm
5. 进入前馈层
6. 再做 residual add

所以这个仓库的 block 不是教学简化版 post-norm，而是更贴近现代 decoder-only 配置。

## 完整 Transformer

`Transformer` 本体的前向路径很直接：

1. token ids 进 embedding
2. 依次通过所有 block
3. 经过最终 RMSNorm
4. 投影回词表 logits

如果没有外部传入位置，它会自动按序列长度构造 position。

## 自定义 Loss 与优化器工具

`CrossEntropyLoss` 会显式把 logits 和 targets reshape 成二维和一维，再做 `log_softmax` 和正确标签索引，最后求平均 NLL。

同一个文件里还定义了：

- `SGDDecay`
- `AdamW`
- `cos_lr_scheduler`
- `gradient_clip`

这也说明仓库的“从零实现”范围不只到前向网络为止，而是把训练最常用的配套工具也一并放进了核心实现层。
