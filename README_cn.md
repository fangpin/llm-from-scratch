[English](./README.md)

# 从零开始的 LLM

本仓库包含一个从零开始实现的现代仅解码器 Transformer 模型，使用 PyTorch 构建，旨在用于教育目的。它包含了现代语言模型的所有基本构建模块，以清晰、模块化和易于理解的方式编写。该项目旨在为学习大型语言模型如何从头开始构建提供全面的资源。

## 特性

*   **从零开始实现：** Transformer 模型的每个组件都使用 PyTorch 从零开始实现，从而深入了解底层机制。
*   **现代架构：** 该模型融合了最先进的语言模型中使用的现代技术，包括：
    *   **RMSNorm：** 用于高效稳定的层归一化。
    *   **SwiGLU：** 前馈网络中的激活函数，可提高性能。
    *   **旋转位置嵌入 (RoPE)：** 用于有效的位置编码。
*   **自定义 BPE 分词器：** 从零开始实现的字节对编码 (BPE) 分词器，可在任何文本语料库上进行训练。
*   **自定义优化器：** 包括 `AdamW` 和 `SGDDecay` 优化器的自定义实现。
*   **全面的训练和生成脚本：** 提供用于在大型语料库上训练模型和使用训练好的模型生成文本的脚本。
*   **全面的测试：** 使用 `pytest` 和快照测试的综合测试套件可确保实现的正确性。

## 已实现的组件

该项目为构建和训练语言模型提供了一个完整的生态系统。关键组件是：

### 核心模型 (`lm/transformer.py`)

*   **`Transformer`**: 组合所有组件的主模型类。
*   **`TransformerBlock`**: Transformer 的单个块，包含多头注意力和前馈网络。
*   **`MultiHeadAttention`**: 多头自注意力机制。
*   **`ScaledDotProductAttention`**: 核心注意力机制。
*   **`FFN`**: 带有 SwiGLU 激活的位置前馈网络。
*   **`RoPE`**: 用于注入位置信息的旋转位置嵌入。
*   **`RmsNorm`**: 均方根层归一化。
*   **`Embedding`**: 令牌嵌入层。
*   **`Linear`**: 自定义线性层。
*   **`Softmax`**: 自定义 softmax 实现。
*   **`CrossEntropyLoss`**: 自定义交叉熵损失函数。

### 分词器 (`lm/bpe_tokenizer.py`)

*   **`BpeTokenizer`**: 从零开始实现的 BPE 分词器。它可以在语料库上进行训练，以学习词汇表和合并。它还支持特殊令牌。

### 訓練和推理

*   **`lm/training.py`**: 用于训练 Transformer 模型的脚本。它包括数据加载、训练循环、验证和检查点。
*   **`lm/generating.py`**: 用于使用经过训练的模型通过 top-p 采样生成文本的脚本。
*   **`lm/checkpoint.py`**: 用于保存和加载模型检查点的实用程序。

### 优化器和实用程序 (`lm/transformer.py`)

*   **`AdamW`**: AdamW 优化器的自定义实现。
*   **`SGDDecay`**: 带有学习率衰减的 SGD 的自定义实现。
*   **`cos_lr_scheduler`**: 带有预热的余弦学习率调度器。
*   **`gradient_clip`**: 用于梯度裁剪的函数。

## 架构

本仓库中的 Transformer 模型是一个仅解码器模型，其架构类似于 GPT 等模型。它专为语言建模任务而设计。关键架构特性是：

*   **预归一化：** 该模型使用 RMSNorm 进行层归一化，该归一化在注意力和前馈层*之前*应用。与后归一化相比，这可以使训练更稳定。
*   **SwiGLU 激活：** 前馈网络使用 SwiGLU (Swish-Gated Linear Unit) 激活函数，该函数已被证明可以提高语言模型的性能。
*   **旋转位置嵌入 (RoPE)：** 该模型使用 RoPE 代替传统的位置嵌入，通过旋转注意力机制中的查询和关键向量来合并位置信息。这是处理长序列的更有效方法。

## 使用方法

### 1. 训练分词器

您可以使用 `lm/bpe_tokenizer.py` 脚本在您自己的文本语料库上训练 BPE 分词器。

```bash
python -m lm.bpe_tokenizer --corpus your_text_file.txt --vocab_size 10000
```

### 2. 准备数据

训练脚本期望训练和验证数据是令牌 ID 的内存映射 numpy 数组的形式。您可以使用训练好的分词器将文本数据转换为这种格式。

### 3. 训练模型

`lm/training.py` 脚本用于训练 Transformer 模型。

```bash
python -m lm.training \
    --train_data path/to/train_data.bin \
    --val_data path/to/val_data.bin \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --vocab_size 10000 \
    --num_layers 6 \
    --max_seq_len 512 \
    --batch_size 32 \
    --iterations 10000 \
    --device cuda:0
```

### 4. 生成文本

一旦你有了训练好的模型，你就可以使用 `lm/generating.py` 来生成文本。

```bash
python -m lm.generating \
    --model_path path/to/your/checkpoint.pt \
    --tokenizer_path path/to/your/tokenizer.json \
    --prompt "Hello, world!" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --device cuda:0
```

## 测试

该项目有一个全面的测试套件，以确保实现的正确性。您可以使用 `pytest` 运行测试：

```bash
pytest
```

测试涵盖：
*   通过将其输出与参考实现进行比较，来测试 Transformer 模型中每个模块的正确性。
*   BPE 分词器的编码和解码，以及其训练过程。
*   优化器和其他实用程序。

## 依赖

*   Python 3.8+
*   PyTorch
*   NumPy
*   einx
*   regex
*   pytest (用于测试)

您可以使用 pip 安装依赖项：
```bash
pip install -r requirements.txt
```

## 许可证

本项目采用 MIT 许可证。有关详细信息，请参阅 `LICENSE` 文件。

## 贡献

欢迎贡献！如果您有任何建议或发现任何错误，请随时提交拉取请求或提出问题。