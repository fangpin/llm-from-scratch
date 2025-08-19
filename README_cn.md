[English](./README.md)

# 从零开始的 Transformer

本仓库包含一个用于教育目的的、从零开始的 PyTorch Transformer 模型实现。它包含了现代 Transformer 架构的所有基本构建模块，代码清晰易懂。

## 特性

*   自定义实现 Transformer 的核心组件。
*   详细且注释良好的代码。
*   包含 RoPE、SwiGLU 和 RMSNorm 等现代技术。
*   自定义优化器 (`SGDDecay`, `AdamW`) 和学习率调度器。

## 已实现的组件

*   **模块**: `Linear`, `Embedding`, `RmsNorm`, `SiLu`, `SwiGlu`, `FFN`, `RoPE`, `Softmax`, `ScaledDotProductAttention`, `MultiHeadAttention`, `TransformerBlock`, `Transformer`, `BpeTokenizer`.
*   **损失函数**: `CrossEntropyLoss`.
*   **优化器**: `SGDDecay`, `AdamW`.
*   **工具**: `cos_lr_scheduler`, `gradient_clip`.

## 架构

本仓库中的 Transformer 模型是一个仅解码器模型，适用于语言建模任务。其特点包括：

*   **预归一化 (Pre-Normalization)**: 在注意力层和前馈网络层之前使用 `RmsNorm` 进行层归一化。
*   **SwiGLU 激活函数**: 前馈网络使用 SwiGLU 激活函数以获得更好的性能。
*   **旋转位置嵌入 (RoPE)**: 通过旋转注意力机制中的查询和键向量来融合位置信息。

## 分词器

本仓库还包含一个在 `lm/bpe_tokenizer.py` 中从零开始实现的字节对编码 (BPE) 分词器。

*   **`BpeTokenizer`**: 一个可以在文本语料库上进行训练的类，用于学习 BPE 词汇表和合并规则。然后，它可以用来将文本编码为词元。

## 使用方法

以下是如何初始化和使用 `Transformer` 模型的简单示例：

```python
import torch
from lm.transformer import Transformer

# 模型参数
d_model = 512
num_heads = 8
d_ff = 2048
vocab_size = 10000
num_layers = 6
max_seq_len = 512

# 创建模型实例
model = Transformer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    vocab_size=vocab_size,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
)

# 创建一个虚拟输入
token_ids = torch.randint(0, vocab_size, (1, max_seq_len))

# 执行前向传播
logits = model(token_ids, train=True)

print(logits.shape)
```

## 安装

要使用此代码，您需要安装 PyTorch 和 einx：

```bash
pip install torch einx
```
