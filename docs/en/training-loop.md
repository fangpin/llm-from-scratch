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

The pretraining path in `llm/training.py` is intentionally compact, but it covers the full execution lifecycle from process-group setup to checkpoint saving. This chapter traces the actual control flow.

## Configuration Boundary

The public entrypoint is `train()`. It pulls all hyperparameters from `llm.args`, asserts that the batch size is divisible by `world_size`, prepares checkpoint and log directories, and then launches:

```python
mp.spawn(_train, args=(args.world_size, args.backend, args), nprocs=args.world_size, join=True)
```

So each rank runs the same `_train()` function with shared configuration and a different rank id.

## Reproducibility Path

`set_random_seed(seed, rank)` derives a rank-specific seed by adding the rank to the global seed, then seeds:

- `torch`
- `torch.cuda`
- `numpy`
- Python `random`

It also forces:

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

This is a practical compromise: per-rank seeds differ, but runs remain reproducible given a fixed world size and launch order.

## Batch Construction

`get_batch()` is the main data loader primitive. It operates on a token-id array `x` and samples random start indices:

```python
ix = torch.randint(0, len(x) - context_length, (batch_size,))
```

For each start index, it creates:

- `input_seqs = x[i : i + context_length]`
- `target_seqs = x[i + 1 : i + 1 + context_length]`

This is the exact autoregressive next-token objective. Inputs and targets differ only by a one-token shift.

Two operational details matter:

- the arrays can be memory-mapped with `mmap_mode="r"`
- the tensors are moved to the requested device before returning

So the training loop never materializes the entire dataset on GPU.

## Process-Group Setup

`_setup_process_group()` owns the multi-GPU runtime bootstrapping:

- sets `MASTER_ADDR=localhost`
- sets `MASTER_PORT=12390`
- sets `NCCL_DEBUG=NONE`
- chooses a local CUDA device as `rank % device_count`
- calls `dist.init_process_group(backend, rank=rank, world_size=world_size)`

This is a local training script rather than a cloud launcher abstraction. The setup is explicit enough that the reader can see exactly how ranks become distributed workers.

`_cleanup_process_group()` then inserts a final barrier before destroying the process group, which prevents ranks from tearing down communication state out of order.

## Model Construction

Inside `_train()`, the script constructs:

```python
model = Transformer(
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    max_seq_len=args.max_seq_len,
    device=device,
)
```

When `world_size > 1`, the model is immediately wrapped in the custom `DDP` implementation from `parallel/ddp.py`.

That wrapper is not cosmetic. It changes how gradients are synchronized later in the loop.

## Optimizer Selection

The script chooses optimizers based on world size:

- single rank: custom `AdamW`
- multi-rank: `ShardedOptimizer(..., AdamW, ...)`

This means the higher-level training logic stays identical while optimizer-state ownership changes under the hood.

The loss is always the custom `CrossEntropyLoss`.

## Validation Path

Validation is only run on rank 0:

1. switch model to eval mode
2. run 100 validation batches
3. average the scalar loss
4. print it
5. log it to TensorBoard
6. switch back to train mode

That choice keeps logging centralized and avoids duplicated validation output from every rank.

The validation path is intentionally simple: no distributed metric reduction, just one reference rank evaluating the current model snapshot.

## Main Training Step

Each iteration performs the following sequence:

1. fetch a training batch
2. forward pass through the model
3. compute cross-entropy loss
4. `optimizer.zero_grad()`
5. `loss.backward()`
6. `gradient_clip(model.parameters(), max_norm=1.0)`
7. if distributed, `model.finish_gradient_sync()`
8. `optimizer.step()`

The explicit `finish_gradient_sync()` call is important. The custom DDP path uses asynchronous bucketed all-reduce, so synchronization is not magically complete at the moment `backward()` returns. The training script makes the completion point visible.

## Learning-Rate Schedule

After the parameter update, the script computes a fresh learning rate with:

```python
cos_lr_scheduler(
    it=i,
    warmup_iters=args.warmup_iters,
    cos_cycle_iters=args.cos_cycle_iters,
    lr_min=args.lr_min,
    lr_max=args.lr_max,
)
```

The resulting value is written into every optimizer param group.

This keeps the schedule external to the optimizer implementation and makes it trivial to inspect or replace.

## Logging

On rank 0, the script writes:

- `loss_train`
- `lr`
- `val_loss`

to `SummaryWriter`, and also prints periodic console logs every `log_interval`.

So the monitoring surface is minimal but sufficient for:

- training stability
- schedule inspection
- basic overfit/divergence detection

## Checkpoint Format

Checkpointing happens every `checkpoint_interval` after iteration 0:

```python
save_checkpoint(model, optimizer, i, path)
```

`llm/checkpoint.py` writes a dictionary containing:

- `model`
- `optimizer`
- `iteration`

This is intentionally small and portable. There is no extra trainer metadata layer.

## Checkpoint Consumption

`llm/generating.py` consumes that format directly:

1. rebuild the same `Transformer`
2. call `load_checkpoint(...)`
3. load tokenizer state
4. encode a prompt
5. repeatedly crop to `model.max_seq_len`
6. compute logits for the last position
7. apply temperature
8. apply top-p truncation
9. sample one token
10. stop on the end-of-text token

This connection is important because it proves the training loop's artifacts are enough for inference without any additional conversion step.

## Distributed Variant in Practice

When `world_size > 1`, two things change operationally:

- each rank uses `mini_batch_size = args.batch_size // world_size`
- model/optimizer synchronization paths become active

But the semantic training objective remains identical: the model still learns next-token prediction over random fixed windows from the same tokenized corpus.

## What the File Optimizes For

`llm/training.py` is not feature-rich by framework standards. It is optimizing for clarity of ownership:

- one function for batch slicing
- one function for process-group setup
- one function per-rank training body
- one top-level launcher
- one checkpoint format

That makes it easy to answer practical implementation questions such as:

- where the dataset enters GPU memory
- when gradients are clipped
- when distributed communication is forced to complete
- when LR changes
- what exactly gets serialized

For a repository that aims to teach model-building mechanics, that directness is more valuable than a larger but more abstract trainer stack.
