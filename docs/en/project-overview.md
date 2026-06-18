---
title: Project Overview
summary: A system map of the repository, from tokenizer and transformer internals to kernels, distributed training, and alignment workflows.
slug: project-overview
locale: en
group: core-stack
order: 1
translationKey: project-overview
sourceFiles:
  - README.md
  - llm/transformer.py
  - llm/bpe_tokenizer.py
  - llm/training.py
  - kernel/flash_attention_triton.py
  - parallel/ddp.py
  - alignment/sft.py
  - alignment/train_rl.py
sourceDocs:
  - docs/1.md
---

# Project Overview

The repository is structured as a teaching-oriented implementation of a modern decoder-only language model in PyTorch. The goal is not to hide the machinery behind a framework wrapper. The goal is to make every major layer inspectable:

- tokenization
- transformer internals
- optimization and training
- kernel acceleration
- distributed execution
- data preparation
- alignment examples

## Module Map

### `llm/`

This is the core model stack:

- `llm/bpe_tokenizer.py` implements a byte-pair tokenizer from scratch
- `llm/transformer.py` implements embeddings, RMSNorm, RoPE, attention, SwiGLU, custom loss, and optimizer utilities
- `llm/training.py` runs training, validation, logging, and checkpointing
- `llm/generating.py` loads a checkpoint and runs top-p text generation
- `llm/checkpoint.py` handles checkpoint save and load

### `kernel/`

This directory isolates attention-kernel optimization work:

- `flash_attention_triton.py` implements the Triton path
- `flash_attention_mock.py` provides a PyTorch-style reference implementation
- `bench_mark/` compares different attention paths and model behaviors

### `parallel/`

This is the custom distributed-training layer:

- `ddp.py` implements gradient synchronization with post-accumulate hooks and bucketed all-reduce
- `sharded_optimizer.py` partitions optimizer ownership across ranks to reduce per-device optimizer-state memory

### `data_processing/`

This is the preprocessing pipeline for large raw text corpora:

- HTML extraction
- language identification
- heuristic filtering
- deduplication
- PII masking
- harmful-content detection
- quality classification

### `alignment/`

This directory contains concrete fine-tuning workflows built around gsm8k and Qwen2.5-Math-1.5B:

- `sft.py` runs supervised fine-tuning
- `train_rl.py` runs reinforcement fine-tuning
- `grpo.py` implements grouped reward normalization and policy-gradient losses
- `drgrpo_grader.py` enforces response format and answer correctness

## Reading Order by Ownership Boundary

If you want to understand the repo from first principles, a practical reading order is:

1. `Tokenizer and Vocabulary`
2. `Transformer Core`
3. `Training Loop and Checkpointing`
4. `Flash Attention and Kernel Optimization`
5. `Distributed Training and Sharded Optimizer`
6. `Data Processing Pipeline`
7. `Supervised Fine-Tuning on gsm8k`
8. `Reinforcement Learning Fine-Tuning on gsm8k`

That order follows the implementation stack more closely than the homepage does.

## What Makes This Repo Useful

Three properties make the project more valuable than a simplified architecture sketch:

1. It implements modern decoder-only building blocks instead of stopping at an outdated baseline.
2. It exposes code paths for both scale work and alignment work, not just the base model.
3. It keeps most pieces small enough that a reader can inspect them directly.

The rest of the docs expand each of those layers with code-adjacent detail.
