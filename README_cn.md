[English](./README.md)

# 从零开始的 LLM

本仓库包含一个用于教育目的的、从零开始的 PyTorch Transformer 模型实现。它包含了现代 Transformer 架构的所有基本构建模块，代码清晰易懂。

## 特性

* 自定义实现 Transformer 的核心组件。
* 详细且注释良好的代码。
* 包含 RoPE、SwiGLU 和 RMSNorm 等现代技术。
* 自定义优化器 (`SGDDecay`, `AdamW`) 和学习率调度器。

## 已实现的组件

* **模块**: `Linear`, `Embedding`, `RmsNorm`, `SiLu`, `SwiGlu`, `FFN`, `RoPE`, `Softmax`, `ScaledDotProductAttention`, `MultiHeadAttention`, `TransformerBlock`, `Transformer`, `BpeTokenizer`.
* **损失函数**: `CrossEntropyLoss`.
* **优化器**: `SGDDecay`, `AdamW`.
* **工具**: `cos_lr_scheduler`, `gradient_clip`.

## 架构

本仓库中的 Transformer 模型是一个仅解码器模型，适用于语言建模任务。其特点包括：

* **预归一化 (Pre-Normalization)**: 在注意力层和前馈网络层之前使用 `RmsNorm` 进行层归一化。
* **SwiGLU 激活函数**: 前馈网络使用 SwiGLU 激活函数以获得更好的性能。
* **旋转位置嵌入 (RoPE)**: 通过旋转注意力机制中的查询和键向量来融合位置信息。

## 文件结构

```
lm/
├── __init__.py
├── bpe_tokenizer.py
├── checkpoint.py
├── generating.py
├── training.py
└── transformer.py
```

* `bpe_tokenizer.py`: 从零开始实现的字节对编码 (BPE) 分词器。现在可以正确处理特殊令牌和各种空白符，包括尾随的空白符，以与 `tiktoken` 等参考分词器的行为保持一致。它还包括一个 `encode_iterable` 方法，用于对大型文本流进行内存高效的分词。`train` 方法也已优化以获得更好的性能。
* `transformer.py`:核心 Transformer 模型，包括所有构建模块，如注意力、FFN 和 RoPE。
* `training.py`: 用于在文本语料库上训练 Transformer 模型的脚本。
* `generating.py`: 用于使用训练好的模型生成文本的脚本。
* `checkpoint.py`: 用于保存和加载模型检查点的实用函数。

## 使用方法

### 1. 训练分词器

`bpe_tokenizer.py` 脚本可用于在您自己的文本数据上训练 BPE 分词器。

```bash
python lm/bpe_tokenizer.py --corpus your_text_file.txt --vocab_size 10000
```

### 2. 训练模型

`training.py` 脚本用于训练 Transformer 模型。

```bash
python lm/training.py \
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

### 3. 生成文本

一旦你有了训练好的模型，你就可以使用 `generating.py` 来生成文本。

```bash
python lm/generating.py \
    --model_path path/to/your/checkpoint.pt \
    --tokenizer_path path/to/your/tokenizer.json \
    --prompt "Hello, world!" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --device cuda:0
```

## 安装

要使用此代码，您需要安装 PyTorch 和 einx：

```bash
pip install torch einx
```

## 先决条件

* Python 3.8 或更高版本
* PyTorch 1.10 或更高版本
* einx
* regex

## 许可证

本项目采用 MIT 许可证。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 引文

如果您在研究中使用此代码，请考虑按如下方式引用：

```bibtex
@misc{transformer-from-scratch,
  author = {Your Name},
  title = {Transformer from Scratch},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/your-username/transformer-from-scratch}},
}
```

## 贡献

欢迎贡献！如果您有任何建议或发现任何错误，请随时提交拉取请求或提出问题。

## 免责声明

此实现仅用于教育目的，可能不适合生产使用。

---

*该 README 由 gemini-cli 自动生成。*

