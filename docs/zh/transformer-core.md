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

整个基础模型栈都集中在 `llm/transformer.py`。这个文件的好处在于“单体但不混乱”：你可以不跳来跳去，就把核心架构、loss 和优化器工具一起读完。

## 文件级职责

`llm/transformer.py` 同时包含三层逻辑：

1. 基础可训练模块，例如 `Linear`、`Embedding`、`RmsNorm`
2. 组合模块，例如 attention、RoPE 和 `TransformerBlock`
3. 训练辅助工具，例如 `CrossEntropyLoss`、`AdamW`、`cos_lr_scheduler` 和 `gradient_clip`

因此主训练脚本可以很短，因为模型局部数学都已经由这个文件负责。

## 基础层

### `Linear`

`Linear` 持有一个形状为 `[out_features, in_features]` 的权重矩阵 `self.w`，前向计算是：

```python
einx.dot("... [in], out [in] -> ... out", x, self.w)
```

初始化时如果没有外部权重，会使用截断正态分布，标准差是：

```python
sigma = math.sqrt(2.0 / (in_features + out_features))
```

这是一个无 bias 的线性投影。去掉 bias 让层更简洁，也和很多 decoder-only 结构的默认选择一致。

### `Embedding`

`Embedding` 持有一张形状为 `[num_embeddings, embedding_dim]` 的参数表，前向时直接用 token id 做索引。

初始化标准差是：

```python
1 / math.sqrt(embedding_dim)
```

这里没有额外的 learned positional embedding，因为位置编码稍后会交给 RoPE。

### `RmsNorm`

`RmsNorm` 的实现完全显式展开：

1. 输入先 cast 到 `float32`
2. 计算最后一维的均方
3. 乘 `rsqrt(variance + eps)`
4. 再乘可学习增益 `g`
5. 最后 cast 回原 dtype

这里有两个实现点值得记住：

- RMSNorm 只按模长归一化，不减均值
- `float32` cast 提高了低精度训练的稳定性

## 非线性与前馈层

### `SiLu`

`SiLu` 直接写成：

```python
torch.sigmoid(x) * x
```

### `SwiGlu`

真正被 block 使用的是 `SwiGlu`。它定义了三层线性变换：

- `w1`：gate 分支
- `w3`：value 分支
- `w2`：投回模型维度

前向公式是：

```python
self.w2(self.silu(self.w1(x)) * self.w3(x))
```

文件里还有这句别名：

```python
FFN = SwiGlu
```

所以所有 block 的前馈层实际上都是 SwiGLU。

## 旋转位置编码

`RoPE` 会在初始化时预先构造 cosine / sine cache。

### cache 的构造方式

构造函数里依次计算：

- 偶数维度对应的 `inv_freq`
- 位置索引 `t = arange(max_seq_len)`
- `freqs = outer(t, inv_freq)`
- 通过 `repeat_interleave` 扩成完整维度
- `cos_cached` 与 `sin_cached`

因此运行时的位置编码变成了查表问题。

### 前向规则

运行时，`RoPE.forward()` 会：

1. 根据 `token_positions` 取出对应的 `cos` 和 `sin`
2. 把最后一维 reshape 成成对坐标
3. 把 `(a, b)` 旋转为 `(-b, a)`
4. 用

```python
x * cos + x_rotated * sin
```

组合原向量与旋转向量

如果输入是 4 维张量，代码会额外 `unsqueeze(1)`，使 trig cache 与 head 维对齐。

## Attention 栈

### `Softmax`

文件自己实现了稳定 softmax：

1. 先减去行最大值
2. 再指数化
3. 用指数和做归一化

这样 attention 逻辑可以完全自洽，不依赖黑箱库函数。

### `ScaledDotProductAttention`

这个模块依次做：

1. `att = QK^T`
2. `att_scale = att / sqrt(d_model)`
3. 如果有 mask，用 `masked_fill(mask, -1e9)` 遮掉位置
4. 对 key 维做 softmax
5. 再乘上 `V`

这里 mask 的语义很明确：`True` 表示该位置不应参与 softmax。

### `MultiHeadAttention`

这个层的执行顺序是：

1. 一次线性投影到 `3 * d_model`
2. 拆成 `Q`、`K`、`V`
3. reshape 成 `[batch, heads, seq, head_dim]`
4. 应用 causal attention
5. 拼回所有 head
6. 再做输出投影

causal mask 在初始化时就缓存好了：

```python
torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
```

运行时只需要按当前 `seq_len` 截取。

### `MultiHeadAttentionWithRoPE`

这个类继承了 `MultiHeadAttention` 的整体流程，但在算 attention score 前插入：

```python
q = self.rope(q, token_positions)
k = self.rope(k, token_positions)
```

如果外部没传 `token_positions`，它会自动构造每个 batch 对应的 `0..seq_len-1`。

所以同一套 attention 模块既能服务训练，也能服务滑动窗口推理。

## Block 组合方式

`TransformerBlock` 是典型的 pre-norm 残差结构：

1. `x_norm = rms_norm1(x)`
2. `x_atten = mult_head_atten(x_norm, token_positions)`
3. `x = x + x_atten`
4. `x_norm = rms_norm2(x)`
5. `x_ffe = ffe(x_norm)`
6. 返回 `x + x_ffe`

这就是一个现代 decoder-only block 的基本形态：

- pre-norm
- attention 内部使用 RoPE
- 前馈层使用 SwiGLU

## 顶层 `Transformer`

顶层模型的前向顺序很简单：

1. token id 先过 `Embedding`
2. 依次通过所有 `TransformerBlock`
3. 经过最终 `RmsNorm`
4. 用 `out_linear` 投影到词表 logits

如果没有显式传位置，模型会根据序列长度自动构造 `token_positions`。

所以顶层接口很小：

```python
forward(token_ids, token_positions=None) -> logits
```

训练脚本和生成脚本都复用了这个接口。

## Loss 实现

`CrossEntropyLoss` 也没有调用黑箱封装，而是显式完成：

1. 把 logits reshape 成二维 `[(...), vocab]`
2. 把 target reshape 成一维 `[(...)]`
3. 计算 `log_softmax`
4. 取出正确标签的 log prob
5. 求负并取平均

这让 next-token loss 的数学定义在文件里一眼就能看到。

## 优化器与调度工具

### `SGDDecay`

`SGDDecay` 是一个教学型优化器，学习率按：

```python
lr / sqrt(t + 1)
```

衰减。

### `AdamW`

自定义 `AdamW` 会显式维护：

- `t`
- 一阶动量 `m`
- 二阶动量 `sm`

每次 step 里会：

1. 更新 `m` 和 `sm`
2. 做 bias correction
3. 用 `m_hat` 与 `sm_hat` 更新参数
4. 再做 decoupled weight decay

这也解释了为什么分布式训练里优化器状态会成为显存大头。

### `cos_lr_scheduler`

这个调度器分三段：

- `warmup_iters` 之前线性升高
- 到 `cos_cycle_iters` 为止做 cosine decay
- 之后固定在 `lr_min`

### `gradient_clip`

`gradient_clip()` 会先计算所有梯度的全局范数，再在超出 `max_norm` 时按统一系数原地缩放。

## 这个文件的架构价值

这个文件最有价值的地方不是“新奇”，而是把现代 decoder-only 的关键选择放在了一个可以完整读完的实现里：

- 无 bias 投影
- RMSNorm
- RoPE
- SwiGLU
- causal self-attention
- 显式 next-token loss
- 显式优化器状态

因此如果你想看“模型内部数学如何直接接到训练循环上”，`llm/transformer.py` 是仓库里最值得精读的一份文件。
