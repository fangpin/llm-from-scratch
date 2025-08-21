[中文](./README_cn.md)

# LLM from Scratch

This repository contains a from-scratch implementation of a modern decoder-only Transformer model in PyTorch, built for educational purposes. It includes all the essential building blocks of a modern language model, written in a clear, modular, and understandable way. The goal of this project is to provide a comprehensive resource for learning how large language models are built from the ground up.

## Features

*   **From-Scratch Implementation:** Every component of the Transformer model is implemented from scratch using PyTorch, providing a deep understanding of the underlying mechanisms.
*   **Modern Architecture:** The model incorporates modern techniques used in state-of-the-art language models, including:
    *   **RMSNorm:** for efficient and stable layer normalization.
    *   **SwiGLU:** activation function in the feed-forward network for improved performance.
    *   **Rotary Position Embeddings (RoPE):** for effective positional encoding.
*   **Custom BPE Tokenizer:** A from-scratch implementation of the Byte Pair Encoding (BPE) tokenizer, which can be trained on any text corpus.
*   **Custom Optimizers:** Includes custom implementations of `AdamW` and `SGDDecay` optimizers.
*   **Comprehensive Training and Generation Scripts:** Provides scripts for training the model on a large corpus and for generating text with a trained model.
*   **Thorough Testing:** A comprehensive test suite using `pytest` and snapshot testing ensures the correctness of the implementation.

## Implemented Components

This project provides a complete ecosystem for building and training a language model. The key components are:

### Core Model (`llm/transformer.py`)

*   **`Transformer`**: The main model class that combines all the components.
*   **`TransformerBlock`**: A single block of the Transformer, containing multi-head attention and a feed-forward network.
*   **`MultiHeadAttention`**: The multi-head self-attention mechanism.
*   **`ScaledDotProductAttention`**: The core attention mechanism.
*   **`FFN`**: The position-wise feed-forward network with SwiGLU activation.
*   **`RoPE`**: Rotary Position Embeddings for injecting positional information.
*   **`RmsNorm`**: Root Mean Square Layer Normalization.
*   **`Embedding`**: The token embedding layer.
*   **`Linear`**: A custom linear layer.
*   **`Softmax`**: A custom softmax implementation.
*   **`CrossEntropyLoss`**: A custom cross-entropy loss function.

### Tokenizer (`llm/bpe_tokenizer.py`)

*   **`BpeTokenizer`**: A from-scratch implementation of the BPE tokenizer. It can be trained on a corpus to learn a vocabulary and merges. It also supports special tokens.

### Training and Inference

*   **`llm/training.py`**: A script for training the Transformer model. It includes data loading, a training loop, validation, and checkpointing.
*   **`llm/generating.py`**: A script for generating text using a trained model with top-p sampling.
*   **`llm/checkpoint.py`**: Utilities for saving and loading model checkpoints.

### Optimizers and Utilities (`llm/transformer.py`)

*   **`AdamW`**: A custom implementation of the AdamW optimizer.
*   **`SGDDecay`**: A custom implementation of SGD with learning rate decay.
*   **`cos_lr_scheduler`**: A cosine learning rate scheduler with warmup.
*   **`gradient_clip`**: A function for gradient clipping.

## Architecture

The Transformer model in this repository is a decoder-only model, similar to the architecture of models like GPT. It is designed for language modeling tasks. The key architectural features are:

*   **Pre-Normalization:** The model uses RMSNorm for layer normalization, which is applied *before* the attention and feed-forward layers. This leads to more stable training compared to post-normalization.
*   **SwiGLU Activation:** The feed-forward network uses the SwiGLU (Swish-Gated Linear Unit) activation function, which has been shown to improve performance in language models.
*   **Rotary Position Embedding (RoPE):** Instead of traditional positional embeddings, this model uses RoPE to incorporate positional information by rotating the query and key vectors in the attention mechanism. This is a more effective way to handle long sequences.

## Usage

### 1. Training the Tokenizer

You can train the BPE tokenizer on your own text corpus using the `llm/bpe_tokenizer.py` script.

```bash
python -m llm.bpe_tokenizer --corpus your_text_file.txt --vocab_size 10000
```

### 2. Preparing the Data

The training script expects the training and validation data to be in the form of memory-mapped numpy arrays of token IDs. You can use the trained tokenizer to convert your text data into this format.

### 3. Training the Model

The `llm/training.py` script is used to train the Transformer model.

```bash
python -m llm.training \
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

### 4. Generating Text

Once you have a trained model, you can use `llm/generating.py` to generate text.

```bash
python -m llm.generating \
    --model_path path/to/your/checkpoint.pt \
    --tokenizer_path path/to/your/tokenizer.json \
    --prompt "Hello, world!" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --device cuda:0
```

## Testing

This project has a comprehensive test suite to ensure the correctness of the implementation. You can run the tests using `pytest`:

```bash
pytest
```

The tests cover:
*   The correctness of each module in the Transformer model by comparing its output with reference implementations.
*   The BPE tokenizer's encoding and decoding, as well as its training process.
*   The optimizers and other utilities.

## Dependencies

*   Python 3.8+
*   PyTorch
*   NumPy
*   einx
*   regex
*   pytest (for testing)

You can install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

---
This README is generated by gemini-cli.
