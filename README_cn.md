<!-- This is a translated version of README.md. Please keep it in sync with the English version. -->
[English](./README.md)

# 从零开始的 LLM

本代码库包含一个从零开始用 PyTorch 实现的现代化仅解码器 Transformer 模型，专为教育目的而构建。它包含了现代语言模型的所有基本构建模块，以清晰、模块化和易于理解的方式编写。该项目的目标是为学习如何从头开始构建大型语言模型提供一个全面的资源。

[我的大模型开源书籍](https://fangpin.github.io/llm-from-scratch/)

## 特性

* **从零开始实现：** Transformer 模型的每个组件都使用 PyTorch 从零开始实现，从而深入理解其底层机制。
* **BPE 分词器：** 从零开始实现的字节对编码 (BPE) 分词器，可以在任何文本语料库上进行训练。
* **现代化架构：** 该模型融合了最先进语言模型中使用的现代技术，包括：
  * **RMSNorm：** 用于高效稳定的层归一化。
  * **SwiGLU：** 前馈网络中的激活函数，以提高性能。
  * **旋转位置嵌入 (RoPE)：** 用于有效的位置编码。
* **数据处理流程:** 一套完整的数据清洗、过滤和预处理工具。
* **Flash Attention 2：** 包含 Flash Attention 2 的 Triton 实现，显著提高了性能和内存效率。
* **分布式训练:** 支持使用分布式数据并行 (DDP) 和分片优化器在多个 GPU 上进行训练。
* **监督微调 (SFT)**: 包含使用对模型Qwen2.5-Math-1.5B在数据集gsm8k上的完整 SFT 示例.
* **强化学习微调 (RLFT)**: 包含使用对模型Qwen2.5-Math-1.5B在数据集gsm8k上的完整 RLFT 示例.

* [从零开始的 LLM](#从零开始的-llm)
  * [特性](#特性)
  * [已实现的组件](#已实现的组件)
    * [核心模型与架构 (`llm/transformer.py`)](#核心模型与架构-llmtransformerpy)
    * [分词器 (`llm/bpe_tokenizer.py`)](#分词器-llmbpe_tokenizerpy)
    * [训练与推理](#训练与推理)
    * [优化器和实用工具 (`llm/transformer.py`)](#优化器和实用工具-llmtransformerpy)
    * [内核优化 (`kernel/`)](#内核优化-kernel)
    * [并行训练 (`parallel/`)](#并行训练-parallel)
    * [数据处理 (`data_processing/`)](#数据处理-data_processing)
  * [使用方法](#使用方法)
    * [1. 准备数据](#1-准备数据)
    * [2. 训练分词器](#2-训练分词器)
    * [3. 训练模型](#3-训练模型)
    * [4. 生成文本](#4-生成文本)
  * [基准测试](#基准测试)
  * [测试](#测试)
  * [训练](#训练)
    * [损失曲线](#损失曲线)
    * [学习率表](#学习率表)
  * [LLM 输出示例](#llm-输出示例)
  * [监督微调](#监督微调)
  * [强化学习微调](#强化学习微调)
  * [在您自己的数据上进行处理和训练](#在您自己的数据上进行处理和训练)
  * [许可证](#许可证)
  * [贡献](#贡献)

## 已实现的组件

该项目为构建和训练语言模型提供了一个完整的生态系统。关键组件包括：

### 核心模型与架构 (`llm/transformer.py`)

* **`Transformer`**：组合所有组件的主模型类。
* **`TransformerBlock`**：Transformer 的单个模块，包含多头注意力和前馈网络。
* **`MultiHeadAttention`**：多头自注意力机制。
* **`ScaledDotProductAttention`**：核心注意力机制。
* **`FFN`**：带有 SwiGLU 激活函数的位置前馈网络。
* **`RoPE`**：用于注入位置信息的旋转位置嵌入。
* **`RmsNorm`**：均方根层归一化。
* **`Embedding`**：词元嵌入层。
* **`Linear`**：自定义线性层。
* **`Softmax`**：自定义 softmax 实现。
* **`CrossEntropyLoss`**：自定义交叉熵损失函数。

本代码库中的 Transformer 模型是一个仅解码器模型，类似于 GPT 等模型的架构。它专为语言建模任务而设计。关键的架构特性是：

* **预归一化：** 模型使用 RMSNorm 进行层归一化，它在注意力和前馈层*之前*应用。与后归一化相比，这能带来更稳定的训练。
* **SwiGLU 激活函数：** 前馈网络使用 SwiGLU (Swish-Gated Linear Unit) 激活函数，该函数已被证明可以提高语言模型的性能。
* **旋转位置嵌入 (RoPE):** 该模型不使用传统的位置嵌入，而是使用 RoPE 通过旋转注意力机制中的查询和键向量来合并位置信息。这是一种更有效处理长序列的方法。

### 分词器 (`llm/bpe_tokenizer.py`)

* **`BpeTokenizer`**：从零开始实现的 BPE 分词器。它可以在语料库上进行训练，以学习词汇表和合并规则。它还支持特殊词元。

### 训练与推理

* **`llm/training.py`**：用于训练 Transformer 模型的脚本。它包括数据加载、训练循环、验证和检查点。
* **`llm/generating.py`**：用于使用训练好的模型通过 top-p 采样生成文本的脚本。
* **`llm/checkpoint.py`**：用于保存和加载模型检查点的实用工具。

### 优化器和实用工具 (`llm/transformer.py`)

* **`AdamW`**：AdamW 优化器的自定义实现。
* **`SGDDecay`**：带有学习率衰减的 SGD 的自定义实现。
* **`cos_lr_scheduler`**：带有预热的余弦学习率调度器。
* **`gradient_clip`**：用于梯度裁剪的函数。

### 内核优化 (`kernel/`)

* **`flash_attention_triton.py`**：Flash Attention 2 的 Triton 实现，以提高性能和内存效率。
* **`flash_attention_mock.py`**：Flash Attention 的模拟实现，用于比较和测试。
* **`bench_mark/`**：一套基准测试，用于比较不同注意力实现和模型组件的性能。

### 并行训练 (`parallel/`)

* **`ddp.py`**：分布式数据并行 (DDP) 的自定义实现，支持跨多个 GPU 的梯度同步，并使用基于桶的通信提高效率。
* **`sharded_optimizer.py`**：参数分片优化器，将模型参数分布在多个设备上，减少内存使用并实现更大模型的训练。

### 数据处理 (`data_processing/`)

该项目包含一套用于预处理大型文本语料库以训练语言模型的工具。这些工具旨在清理、过滤和准备数据，以提高训练模型的质量。关键的数据处理步骤包括：

* **`html_process.py`**: 从 HTML 内容中提取纯文本。这对于处理像 Common Crawl 这样的网络抓取数据非常有用。
* **`language_identification.py`**: 识别给定文本的语言。这可以用于筛选特定语言的文本。
* **`quality_filter.py`**: 一套启发式过滤器，用于删除低质量内容，例如词数、平均词长和字母字符比例的过滤器。
* **`deduplicate.py`**: 提供精确的逐行去重和使用 MinHash 进行近似去重的功能。
* **`mask_pii.py`**: 屏蔽个人身份信息 (PII)，如电子邮件地址、电话号码和 IP 地址。
* **`harmful_detect.py`**: 使用预训练的 FastText 模型检测有害内容，包括 NSFW 和有毒语言。
* **`quality_classfier.py`**: 一个基于 FastText 的分类器，用于区分高质量和低质量内容。

## 使用方法

**注意:** 以下命令使用 `uv run`，这是一个在虚拟环境中运行命令的工具。如果您没有使用 `uv`，您可以将 `uv run` 替换为 `python`。例如，`uv run -m llm.training` 变为 `python -m llm.training`。

### 1. 准备数据

训练脚本期望训练和验证数据是内存映射的 NumPy 数组形式的词元 ID。您可以使用训练好的分词器将您的文本数据转换为此格式。

通过以下方式下载数据

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 2. 训练分词器

您可以使用 `llm/bpe_tokenizer.py` 脚本在您自己的文本语料库上训练 BPE 分词器。

准备用于训练的词元 ID。如果您有多个文件作为训练语料库，只需使用特殊词元 "<|endoftext|>" 将这些文件合并即可。

```bash

uv run -m llm.bpe_tokenizer
```

### 3. 训练模型

使用 `llm/training.py` 脚本来训练 Transformer 模型。

```bash
uv run -m llm.training
```

要进行分布式训练，请使用以下命令：

```bash
uv run -m llm.training --world_size 6 --batch_size 768
```

### 4. 生成文本

一旦您有了训练好的模型，就可以使用 `llm/generating.py` 来生成文本。

```bash
uv run -m llm.generating
```

## 基准测试

关于模型性能和组件的基准测试详情，请参阅 [BENCHMARK.md](BENCHMARK.md)。

## 项目主页

仓库内包含一个用于 GitHub Pages 的 Astro 双语项目站，位于 `site/`。

* [项目主页](https://fangpin.github.io/llm-from-scratch/)
* [文档首页](https://fangpin.github.io/llm-from-scratch/docs/)
* [中文文档首页](https://fangpin.github.io/llm-from-scratch/zh/docs/)
* [Transformer 核心章节](https://fangpin.github.io/llm-from-scratch/docs/transformer-core/)
* [分布式训练章节](https://fangpin.github.io/llm-from-scratch/docs/distributed-training/)

```bash
npm --prefix site install
npm --prefix site run dev
```

## 测试

该项目有一个全面的测试套件，以确保实现的正确性。您可以使用 `pytest` 运行测试：

```bash
uv run pytest
```

测试覆盖范围：

* 通过将其输出与参考实现进行比较，来验证 Transformer 模型中每个模块的正确性。
* BPE 分词器的编码和解码，以及其训练过程。
* 优化器和其他实用工具。
* 分布式训练设置。

## 训练

### 损失曲线

![损失曲线](img/loss.png)

### 学习率表

![学习率表](img/lr.png)

## LLM 输出示例

在训练完 Tiny stories 数据集后，您可以使用训练好的模型，通过提示“tell you a story”来生成文本，可以得到以下输出。

```bash
Prompt: tell you a story
Completion:  about an a magic box. It said: "I know you can live there, and you can choose. You will see it and keep it in your heart. It will be fun and healthy."
Lily was amazed. She liked the heart. She liked the story. She wondered what the heart was. She wondered what was inside. She wanted to find out what the heart was.
"Please, Mr. Snowman. He is a gift from my story. He is very special. He is very special. He has a new heart and a smile. He is a symbol. He is a gift from his grandma. He is very proud of him. He wanted to be his friend. He took the heart and went to his room. He told Lily he was very smart and kind.
Lily was happy. She had made a new friend. She did not know that Mr. Snowman was a good friend. He had a very special heart. He had a friend. He had a heart and a hug. He could tell Lily about his heart. He had many friends. He did not hear any of the heart. He was a big, friendly dog. He liked to play with Lily. He liked to play with Lily. He had many friends.
<|endoftext|>

```

```bash
Prompt: tell you a story
Completion: ."
Tim and Sam looked at each other and started to laugh. They knew they were going to have a big party. They said sorry to each other and hugged. They played games and ate cake and shared their cookies. They were happy and loved.
<|endoftext|>
```

## 监督微调

使用监督式微调 (SFT) 在 gsm8k 数据集上对 Qwen2.5-Math-1.5B 模型进行了微调。结果如下：

* **零样本准确率:** 从 1.56% 提升到 62.9%。
* **输出格式遵循率:** 从 18.9% 提升到 100%。

**1. 获取 gsm8k 数据集**

```bash
cd dataset
# 下载训练集
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
# 下载测试集
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

**2. 运行评估和微调**

在 gsm8k 数据集上评估 Qwen/Qwen2.5-Math-1.5B 模型的零样本准确率和格式遵循度：

```bash
uv run -m alignment.evaluate
```

在 gsm8k 数据集上对 Qwen/Qwen2.5-Math-1.5B 模型进行 SFT 微调，并测试微调后的推理准确率和格式遵循度：

```bash
uv run -m alignment.sft
```

**3. 微调后的示例输出**

Prompt:

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Assistant: <think>
```

## 强化学习微调

在监督微调（SFT）之后，您可以使用强化学习（RL）来进一步使模型与特定目标对齐，例如提高数学推理的准确性。本项目实现了 GRPO（Group-wise Reward Policy Optimization），一种类似 PPO 的算法，以根据奖励信号对模型进行微调。

要以 SFT 模型为起点，在 gsm8k 数据集上开始 RL 微调过程，请运行以下命令：

```bash
uv run -m alignment.train_rl
```

## 在您自己的数据上进行处理和训练

通过以下方式下载示例 Common Crawl 数据：

```bash
mkdir -p data && cd data
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
```

## 许可证

该项目根据 MIT 许可证授权。有关详细信息，请参阅 `LICENSE` 文件。

## 贡献

欢迎贡献！如果您有任何建议或发现任何错误，请随时提交拉取请求或开启一个 issue。
