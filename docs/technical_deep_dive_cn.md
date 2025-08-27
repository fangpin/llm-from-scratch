# 源码解析：从零开始构建一个现代 LLM——`llm-from-scratch` 项目深度剖析

## 前言

在上一篇文章中，我们向大家推荐了 [`llm-from-scratch`](https://github.com/fangpin/llm-from-scratch) 这个宝藏级的开源项目，并介绍了它的核心特性和使用方法。今天，我们将更进一步，深入项目的源码，从技术的角度，剖析一个现代的大语言模型是如何从一行行代码中诞生的。

本文的目标读者是对 LLM 技术有一定了解，并希望通过阅读源码来加深理解的开发者和学习者。准备好了吗？让我们开始这场激动人心的代码之旅吧！

## 设计哲学：为何选择“从零开始”？

在 `transformers` 库已经成为事实标准的今天，为何我们还要选择“从零开始”？

[`llm-from-scratch`](https://github.com/fangpin/llm-from-scratch) 的核心设计哲学，就是 **“为了学习而构建”**。它刻意避免了对高级库的过度封装，旨在将现代 LLM 的核心组件以最清晰、最直接的方式呈现给学习者。

项目选型紧跟前沿，没有历史包袱，直接采用了 Llama 等现代模型验证过的优秀设计，如 RoPE、SwiGLU 和 RMSNorm。这使得它成为一个学习现代 LLM 架构的绝佳起点。

---

## 核心模块源码剖析

我们将按照模型的数据流，从输入到输出，逐一剖析项目的核心模块。

### 1. `llm.bpe_tokenizer`：一切的起点

模型处理的第一步是分词。`bpe_tokenizer.py` 中实现了一个 BPE 分词器。其核心思想是通过迭代地合并最高频的字节对来构建词汇表。

```python
# llm/bpe_tokenizer.py

class BpeTokenizer:
    # ...
    def train(self, text: str, vocab_size: int):
        # ...
        while len(self.merges) < num_merges:
            # 找出频率最高的字节对
            pair_counts = self._count_pair_freq(tokens)
            if not pair_counts:
                break
            p = max(pair_counts, key=pair_counts.get)
            # ...
            # 合并字节对
            tokens = self._merge(tokens, p, new_id)
            self.merges[p] = new_id
            # ...
```

这个训练过程虽然简洁，但完整地展示了 BPE 算法的精髓。通过这个模块，你可以亲手训练一个专属于你的数据集的分词器。

### 2. `llm.transformer`：模型的心脏

这是整个项目的核心。我们将重点剖析 `Transformer` 类的关键组成部分。

#### `RmsNorm`：更高效的归一化

在深入 Transformer 模块之前，我们先来看看这个被现代 LLM 广泛采用的归一化层。

```python
# llm/transformer.py

class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        input_dtype = x.dtype
        x = x.to(torch.float32)
        # 计算均方根的倒数
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # 乘以可学习的 gain 参数
        return (self.g * x).to(input_dtype)
```

相比于 `LayerNorm`，`RmsNorm` 省去了计算均值和减去均值的操作，也只有一个可学习的 `gain` 参数（`self.g`），而没有 `bias` 参数。这使得它的计算量更小，效率更高，同时在实践中被证明同样有效。

#### `RoPE`：优雅地注入位置信息

旋转位置编码（Rotary Position Embedding）是近年来 LLM 领域的一大创新。它的核心思想是，对于一个词向量，其位置信息可以通过一个旋转矩阵来表示。

```python
# llm/transformer.py

class RoPE(torch.nn.Module):
    # ...
    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Float[Tensor, "... seq"]) -> torch.Tensor:
        # 根据 token 的位置，从缓存中取出对应的 cos 和 sin 值
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 将词向量两两一组进行旋转
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x_rotated = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)
        x_rotated = x_rotated.view(*x.shape)

        # 应用旋转
        x_rot = x * cos + x_rotated * sin
        return x_rot
```

`RoPE` 的实现非常巧妙。它预先计算并缓存了不同位置对应的 `cos` 和 `sin` 值。在 `forward` 方法中，它将输入的词向量 `x` 两两一组，看作是复数 `a+ib`，然后乘以 `cos + i*sin`，从而实现旋转。这种方式将绝对位置编码和相对位置编码有机地结合了起来。

#### `SwiGlu`：带门控的激活函数

`SwiGlu` 是 `Glu` (Gated Linear Unit) 的一种变体，它在 `FFN` (Feed-Forward Network) 中表现出色。

```python
# llm/transformer.py

class SwiGlu(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_in, d_hidden, device=device, dtype=dtype)
        self.w3 = Linear(d_in, d_hidden, device=device, dtype=dtype)
        self.w2 = Linear(d_hidden, d_out, device=device, dtype=dtype)
        self.silu = SiLu() # SiLU(x) = x * sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w3(x) 作为门控，控制 w1(x) 的信息流
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
```

`SwiGlu` 的关键在于 `self.silu(self.w1(x)) * self.w3(x)`。这里，`self.w3(x)` 的输出被用作一个“门”，它乘以 `self.silu(self.w1(x))` 的输出，从而动态地控制了哪些信息可以通过。这种机制被认为能够帮助模型更好地学习复杂的模式。

#### `TransformerBlock`：组装核心

`TransformerBlock` 将上述组件组装在一起，构成了 Transformer 的核心单元。

```python
# llm/transformer.py

class TransformerBlock(torch.nn.Module):
    # ...
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-Normalization
        x_norm = self.rms_norm1(x)
        # Multi-Head Attention with RoPE
        x_atten = self.mult_head_atten(x_norm, token_positions)
        # Residual Connection
        x = x + x_atten

        # Pre-Normalization
        x_norm = self.rms_norm2(x)
        # Feed-Forward Network
        x_ffe = self.ffe(x_norm)
        # Residual Connection
        return x + x_ffe
```

代码清晰地展示了“Pre-Normalization”的结构：每个子层（Attention 和 FFN）的输入都先经过 `RmsNorm`，然后通过残差连接（Residual Connection）与子层的输出相加。这是目前主流的 Transformer 实现方式。

### 3. `llm.training`：让模型学会思考

`training.py` 脚本负责模型的训练。

```python
# llm/training.py

def train():
    # ...
    for i in range(args.iterations + 1):
        # 1. 学习率调度
        lr = cos_lr_scheduler(...)
        # ...

        # 2. 获取数据批次
        inputs, targets = get_batch(train_data, ...)

        # 3. 前向传播和计算损失
        logits = model(inputs)
        loss = criterion(logits, targets)

        # 4. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), max_norm=1.0)
        optimizer.step()
        # ...
```

训练循环的逻辑非常清晰：
1.  **学习率调度**：使用 `cos_lr_scheduler` 动态调整学习率。
2.  **数据获取**：`get_batch` 函数通过 `numpy.memmap` 高效地从磁盘读取数据，避免了内存瓶颈。
3.  **前向/后向传播**：标准的 PyTorch 训练流程。
4.  **梯度裁剪**：`gradient_clip` 用于防止梯度爆炸，保证训练的稳定性。

## 如何参与贡献？

[`llm-from-scratch`](https://github.com/fangpin/llm-from-scratch) 是一个充满活力的开源项目，欢迎所有感兴趣的开发者参与贡献。你可以通过以下方式参与：

-   **报告问题 (Issues):** 发现 Bug？或者有新的功能建议？欢迎提交 Issue。
-   **提交拉取请求 (Pull Requests):** 无论是修复 Bug、完善文档，还是实现新功能，我们都欢迎你的 PR。
-   **分享你的经验:** 将你学习和使用这个项目的经验分享给更多的人。

## 总结

通过对 [`llm-from-scratch`](https://github.com/fangpin/llm-from-scratch) 源码的深度剖析，我们不仅能够理解一个现代 LLM 的工作原理，更能够欣赏到其背后优雅而巧妙的设计思想。

这个项目为我们提供了一个绝佳的平台，让我们能够站在巨人的肩膀上，看得更远。希望这篇文章能够帮助你更好地理解这个项目，并激发你深入探索 LLM 世界的兴趣。

再次感谢 [`llm-from-scratch`](https://github.com/fangpin/llm-from-scratch) 的作者和贡献者们！

---