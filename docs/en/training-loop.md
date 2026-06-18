---
title: Training Loop and Checkpointing
summary: How llm/training.py slices data, initializes distributed execution, runs validation, applies schedules, and saves checkpoints.
slug: training-loop
locale: en
group: core-stack
order: 4
translationKey: training-loop
sourceFiles:
  - llm/training.py
  - llm/checkpoint.py
  - llm/generating.py
  - llm/args.py
sourceDocs:
  - docs/1.md
---

# Training Loop and Checkpointing

The main pretraining path lives in `llm/training.py`. It owns much more than a single `for` loop. The file coordinates:

- random seeds
- data loading
- device and process-group setup
- model construction
- optimizer selection
- validation cadence
- gradient synchronization
- learning-rate scheduling
- checkpointing

## Batch Construction

`get_batch()` samples random starting offsets from a memory-mapped token-id array and creates:

- `input_seqs = x[i : i + context_length]`
- `target_seqs = x[i + 1 : i + 1 + context_length]`

That means the model is always trained on next-token prediction over fixed windows.

The function returns tensors already moved onto the requested device, so the training loop does not need a second device-transfer step.

## Process-Group Setup

`_setup_process_group()` configures NCCL, master address, and master port, then picks the local CUDA device from the rank modulo the number of visible devices.

That setup is intentionally explicit:

- `MASTER_ADDR=localhost`
- `MASTER_PORT=12390`
- `backend` defaults to `nccl`

The training script is built for local multi-GPU execution rather than a hidden launcher environment.

## Model and Optimizer Initialization

The script creates a `Transformer` from `llm/transformer.py` using CLI hyperparameters from `llm/args.py`.

When `world_size > 1`:

- the model is wrapped by the custom `DDP`
- the optimizer becomes `ShardedOptimizer(..., AdamW, ...)`

When `world_size == 1`:

- the model stays local
- the optimizer is the local custom `AdamW`

So the same script can cover both single-device training and distributed training with a shared high-level loop.

## Validation Loop

Every `val_interval`, rank 0:

1. switches the model to eval mode
2. runs 100 validation batches
3. averages the loss
4. logs it to TensorBoard
5. switches back to train mode

That validation path is deliberately narrow but practical. It gives a consistent training-health signal without overcomplicating the script.

## Training Step

Each iteration performs:

1. sample a training batch
2. forward pass to logits
3. compute custom cross-entropy loss
4. zero gradients
5. backward pass
6. gradient clipping
7. distributed gradient synchronization if enabled
8. optimizer step

The distributed sync call is explicit:

```python
if args.world_size > 1:
    model.finish_gradient_sync()
```

That line is where the custom DDP path is forced to finish any asynchronous bucket communication before parameter updates happen.

## Learning-Rate Schedule

`cos_lr_scheduler()` implements:

- linear warmup until `warmup_iters`
- cosine decay until `cos_cycle_iters`
- `lr_min` afterwards

The script recalculates the learning rate every iteration and writes it back into each optimizer param group.

That keeps scheduling logic outside the optimizer itself and easy to inspect.

## Checkpointing

`save_checkpoint()` stores:

- model state dict
- optimizer state dict
- iteration number

The training loop saves checkpoints every `checkpoint_interval`, after iteration 0.

`load_checkpoint()` in `llm/checkpoint.py` restores model state and optionally optimizer state, then returns the saved iteration number.

## Generation Path

`llm/generating.py` is the simplest downstream consumer of this training output:

1. construct the same `Transformer`
2. load a checkpoint
3. load the tokenizer
4. encode a prompt
5. autoregressively sample new tokens with temperature and top-p filtering
6. stop at the end-of-text token

That file matters in the docs because it proves the training artifacts are not just saved, they are directly consumable by a lightweight inference path inside the same repo.

## Operational Summary

The training loop is intentionally plain:

- mmap-based data loading
- one model constructor
- one validation loop
- one scheduler
- one checkpoint format

That simplicity is the point. The repo is trying to make the training lifecycle legible, not abstract it away.
