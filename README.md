[中文](./README_cn.md)

# Transformer from Scratch

This repository contains a from-scratch implementation of a Transformer model in PyTorch, built for educational purposes. It includes all the essential building blocks of a modern transformer architecture, written in a clear and understandable way.

## Features

*   Custom implementations of core transformer components.
*   Detailed and well-commented code.
*   Includes modern techniques like RoPE, SwiGLU, and RMSNorm.
*   Custom optimizers (`SGDDecay`, `AdamW`) and a learning rate scheduler.

## Implemented Components

*   **Modules**: `Linear`, `Embedding`, `RmsNorm`, `SiLu`, `SwiGlu`, `FFN`, `RoPE`, `Softmax`, `ScaledDotProductAttention`, `MultiHeadAttention`, `TransformerBlock`, `Transformer`, `BpeTokenizer`.
*   **Loss Function**: `CrossEntropyLoss`.
*   **Optimizers**: `SGDDecay`, `AdamW`.
*   **Utilities**: `cos_lr_scheduler`, `gradient_clip`.

## Architecture

The transformer model in this repository is a decoder-only model, suitable for language modeling tasks. It features:

*   **Pre-Normalization**: Uses `RmsNorm` for layer normalization before the attention and feed-forward layers.
*   **SwiGLU Activation**: The feed-forward network uses the SwiGLU activation function for better performance.
*   **Rotary Position Embedding (RoPE)**: Incorporates positional information by rotating the query and key vectors in the attention mechanism.

## Tokenizer

This repository also includes a from-scratch implementation of a Byte Pair Encoding (BPE) tokenizer in `lm/bpe_tokenizer.py`.

*   **`BpeTokenizer`**: A class that can be trained on a text corpus to learn a BPE vocabulary and merges. It can then be used to encode text into tokens.

## Usage

Here is a simple example of how to initialize and use the `Transformer` model:

```python
import torch
from lm.transformer import Transformer

# Model parameters
d_model = 512
num_heads = 8
d_ff = 2048
vocab_size = 10000
num_layers = 6
max_seq_len = 512

# Create a model instance
model = Transformer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    vocab_size=vocab_size,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
)

# Create a dummy input
token_ids = torch.randint(0, vocab_size, (1, max_seq_len))

# Perform a forward pass
logits = model(token_ids, train=True)

print(logits.shape)
```

## Installation

To use this code, you will need to have PyTorch and einx installed:

```bash
pip install torch einx
```