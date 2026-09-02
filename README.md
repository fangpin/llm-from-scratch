[中文](./README_cn.md)

# LLM from Scratch

This repository contains a from-scratch implementation of a modern decoder-only Transformer model in PyTorch, built for educational purposes. It includes all the essential building blocks of a modern language model, written in a clear, modular, and understandable way. The goal of this project is to provide a comprehensive resource for learning how large language models are built from the ground up.

[My open source book for LLM](https://fangpin.github.io/llm-from-scratch/)

## Features

* **From-Scratch Implementation:** Every component of the Transformer model is implemented from scratch using PyTorch, providing a deep understanding of the underlying mechanisms.
* **BPE Tokenizer:** A from-scratch implementation of the Byte Pair Encoding (BPE) tokenizer, which can be trained on any text corpus.
* **Modern Architecture:** The model incorporates modern techniques used in state-of-the-art language models, including:
  * **RMSNorm:** for efficient and stable layer normalization.
  * **SwiGLU:** activation function in the feed-forward network for improved performance.
  * **Rotary Position Embeddings (RoPE):** for effective positional encoding.
* **Data Processing Pipeline:** A comprehensive suite of tools for data cleaning, filtering, and pre-processing.
* **Flash Attention 2:** Includes a Triton implementation of Flash Attention 2, significantly improving performance and memory efficiency.
* **Distributed Training:** Supports Distributed Data Parallel (DDP) and sharded optimizer for training on multiple GPUs.
* **Supervised Fine-tuning (SFT)**: Containing fully SFT example on model Qwen2.5-Math-1.5B and dataset gsm8k.
* **Reinforcement Learning Fine-tuning (RLFT)**: Containing fully RLFT example on model Qwen2.5-Math-1.5B and dataset gsm8k.

* [LLM from Scratch](#llm-from-scratch)
  * [Features](#features)
  * [Implemented Components](#implemented-components)
    * [Core Model \& Architecture (`llm/transformer.py`)](#core-model--architecture-llmtransformerpy)
    * [Tokenizer (`llm/bpe_tokenizer.py`)](#tokenizer-llmbpe_tokenizerpy)
    * [Training and Inference](#training-and-inference)
    * [Optimizers and Utilities (`llm/transformer.py`)](#optimizers-and-utilities-llmtransformerpy)
    * [Kernel Optimizations (`kernel/`)](#kernel-optimizations-kernel)
    * [Parallel Training (`parallel/`)](#parallel-training-parallel)
    * [Data Processing (`data_processing/`)](#data-processing-data_processing)
  * [Usage](#usage)
    * [1. Preparing the Data](#1-preparing-the-data)
    * [2. Training the Tokenizer](#2-training-the-tokenizer)
    * [3. Training the Model](#3-training-the-model)
    * [4. Generating Text](#4-generating-text)
  * [Benchmarking](#benchmarking)
  * [Testing](#testing)
  * [Training](#training)
    * [Loss Curve](#loss-curve)
    * [Learning Rate Schedule](#learning-rate-schedule)
  * [Example LLM Output](#example-llm-output)
  * [Supervised Fine-tuning](#supervised-fine-tuning)
  * [Reinforcement Learning Fine-tuning](#reinforcement-learning-fine-tuning)
  * [Processing \& training on your own data](#processing--training-on-your-own-data)
  * [License](#license)
  * [Contributing](#contributing)

## Implemented Components

This project provides a complete ecosystem for building and training a language model. The key components are:

### Core Model & Architecture (`llm/transformer.py`)

* **`Transformer`**: The main model class that combines all the components.
* **`TransformerBlock`**: A single block of the Transformer, containing multi-head attention and a feed-forward network.
* **`MultiHeadAttention`**: The multi-head self-attention mechanism.
* **`ScaledDotProductAttention`**: The core attention mechanism.
* **`FFN`**: The position-wise feed-forward network with SwiGLU activation.
* **`RoPE`**: Rotary Position Embeddings for injecting positional information.
* **`RmsNorm`**: Root Mean Square Layer Normalization.
* **`Embedding`**: The token embedding layer.
* **`Linear`**: A custom linear layer.
* **`Softmax`**: A custom softmax implementation.
* **`CrossEntropyLoss`**: A custom cross-entropy loss function.

The Transformer model in this repository is a decoder-only model, similar to the architecture of models like GPT. It is designed for language modeling tasks. The key architectural features are:

* **Pre-Normalization:** The model uses RMSNorm for layer normalization, which is applied *before* the attention and feed-forward layers. This leads to more stable training compared to post-normalization.
* **SwiGLU Activation:** The feed-forward network uses the SwiGLU (Swish-Gated Linear Unit) activation function, which has been shown to improve performance in language models.
* **Rotary Position Embedding (RoPE):** Instead of traditional positional embeddings, this model uses RoPE to incorporate positional information by rotating the query and key vectors in the attention mechanism. This is a more effective way to handle long sequences.

### Tokenizer (`llm/bpe_tokenizer.py`)

* **`BpeTokenizer`**: A from-scratch implementation of the BPE tokenizer. It can be trained on a corpus to learn a vocabulary and merges. It also supports special tokens.

### Training and Inference

* **`llm/training.py`**: A script for training the Transformer model. It includes data loading, a training loop, validation, and checkpointing.
* **`llm/generating.py`**: A script for generating text using a trained model with top-p sampling.
* **`llm/checkpoint.py`**: Utilities for saving and loading model checkpoints.

### Optimizers and Utilities (`llm/transformer.py`)

* **`AdamW`**: A custom implementation of the AdamW optimizer.
* **`SGDDecay`**: A custom implementation of SGD with learning rate decay.
* **`cos_lr_scheduler`**: A cosine learning rate scheduler with warmup.
* **`gradient_clip`**: A function for gradient clipping.

### Kernel Optimizations (`kernel/`)

* **`flash_attention_triton.py`**: A Triton implementation of Flash Attention 2 for improved performance and memory efficiency.
* **`flash_attention_mock.py`**: A mock implementation of flash attention for comparison and testing.
* **`bench_mark/`**: A suite of benchmarks to compare the performance of different attention implementations and model components.

### Parallel Training (`parallel/`)

* **`ddp.py`**: A custom implementation of Distributed Data Parallel (DDP) that supports gradient synchronization across multiple GPUs with bucket-based communication for efficiency.
* **`sharded_optimizer.py`**: A parameter-sharded optimizer that distributes model parameters across multiple devices, reducing memory usage and enabling training of larger models.

### Data Processing (`data_processing/`)

This project includes a suite of tools for pre-processing large text corpora for training language models. These tools are designed to clean, filter, and prepare the data to improve the quality of the trained model. The key data processing steps are:

* **`html_process.py`**: Extracts plain text from HTML content. This is useful for processing web-scraped data like Common Crawl.
* **`language_identification.py`**: Identifies the language of a given text. This can be used to filter for specific languages.
* **`quality_filter.py`**: A set of heuristic filters to remove low-quality content, such as filters for word count, average word length, and the proportion of alphabetic characters.
* **`deduplicate.py`**: Provides functions for both exact line-by-line deduplication and near-duplicate detection using MinHash.
* **`mask_pii.py`**: Masks personally identifiable information (PII) such as email addresses, phone numbers, and IP addresses.
* **`harmful_detect.py`**: Detects harmful content, including NSFW and toxic language, using pre-trained FastText models.
* **`quality_classfier.py`**: A FastText-based classifier to distinguish between high-quality and low-quality content.

## Usage

**Note:** The following commands use `uv run`, which is a tool for running commands in a virtual environment. If you are not using `uv`, you can replace `uv run` with `python`. For example, `uv run -m llm.training` becomes `python -m llm.training`.

### 1. Preparing the Data

The training script expects the training and validation data to be in the form of memory-mapped numpy arrays of token IDs. You can use the trained tokenizer to convert your text data into this format.

Downloading data by

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

### 2. Training the Tokenizer

You can train the BPE tokenizer on your own text corpus using the `llm/bpe_tokenizer.py` script.

Preparing token ids for training. If you have multiple files as training corpus, just simply merge these files by a special token "<|endoftext|>"

```bash

uv run -m llm.bpe_tokenizer
```

### 3. Training the Model

The `llm/training.py` script is used to train the Transformer model.

```bash
uv run -m llm.training
```

To run with distributed training, use the following command:

```bash
uv run -m llm.training --world_size 6 --batch_size 768
```

### 4. Generating Text

Once you have a trained model, you can use `llm/generating.py` to generate text.

```bash
uv run -m llm.generating
```

## Benchmarking

For details on model performance and component benchmarks, please see [BENCHMARK.md](BENCHMARK.md).

## Project Site

The repository includes a bilingual Astro project site for GitHub Pages in `site/`.

* [Project Site](https://fangpin.github.io/llm-from-scratch/)
* [Docs overview](https://fangpin.github.io/llm-from-scratch/docs/)
* [Chinese docs overview](https://fangpin.github.io/llm-from-scratch/zh/docs/)
* [Transformer Core chapter](https://fangpin.github.io/llm-from-scratch/docs/transformer-core/)
* [Distributed Training chapter](https://fangpin.github.io/llm-from-scratch/docs/distributed-training/)

```bash
npm --prefix site install
npm --prefix site run dev
```

## Testing

This project has a comprehensive test suite to ensure the correctness of the implementation. You can run the tests using `pytest`:

```bash
uv run pytest
```

The tests cover:

* The correctness of each module in the Transformer model by comparing its output with reference implementations.
* The BPE tokenizer's encoding and decoding, as well as its training process.
* The optimizers and other utilities.
* Distributed training setup.

## Training

### Loss Curve

![Loss Curve](img/loss.png)

### Learning Rate Schedule

![Learning Rate Schedule](img/lr.png)

## Example LLM Output

After training Tiny stories dataset, you can get the following output by using the trained model to generate text with the prompt "tell you a story".

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

## Supervised Fine-tuning

I fine-tuned the Qwen2.5-Math-1.5B model on the gsm8k dataset using Supervised Fine-Tuning (SFT). The results are as follows:

* **Zero-shot accuracy:** Increased from 1.56% to 62.9%.
* **Output format compliance:** Increased from 18.9% to 100%.

**1. Getting the gsm8k dataset**

```bash
cd dataset
# Download the training set
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
# Download the test set
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

**2. Running Evaluation and Fine-tuning**

To evaluate the zero-shot accuracy and format compliance of the Qwen/Qwen2.5-Math-1.5B model on the gsm8k dataset:

```bash
uv run -m alignment.evaluate
```

To perform SFT fine-tuning on the Qwen/Qwen2.5-Math-1.5B model on the gsm8k dataset and test the post-fine-tuning inference accuracy and format compliance:

```bash
uv run -m alignment.sft
```

**3. Example Output after Fine-tuning**

Prompt:

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Assistant: <think>
```

## Reinforcement Learning Fine-tuning

After Supervised Fine-tuning (SFT), you can further align the model with a specific objective, such as improving mathematical reasoning accuracy, using Reinforcement Learning (RL). This project implements GRPO (Group-wise Reward Policy Optimization), a PPO-like algorithm, to fine-tune the model based on a reward signal.

To start the RL fine-tuning process on the gsm8k dataset using the SFT model as a starting point, run the following command:

```bash
uv run -m alignment.train_rl
```

## Processing & training on your own data

Downloading example Common Crawl data by:

```bash
mkdir -p data && cd data
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.
