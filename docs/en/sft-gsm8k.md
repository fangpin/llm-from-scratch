---
title: Supervised Fine-Tuning on gsm8k
summary: How alignment/sft.py turns gsm8k examples into prompt-completion training data and measures accuracy plus format compliance.
slug: sft-gsm8k
locale: en
group: alignment-workflows
order: 8
translationKey: sft-gsm8k
sourceFiles:
  - alignment/sft.py
  - alignment/dataset.py
  - alignment/r1_prompt.py
  - alignment/evaluate.py
  - alignment/drgrpo_grader.py
  - alignment/args.py
sourceDocs:
  - docs/5-sft.md
---

# Supervised Fine-Tuning on gsm8k

The repo's SFT example is not a generic "alignment" sketch. It is a concrete workflow around:

- `Qwen/Qwen2.5-Math-1.5B`
- `gsm8k`
- explicit output-format constraints

The main implementation lives in `alignment/sft.py`.

## Data Construction

`alignment/dataset.py` defines `Gsm8kDataset`, which reads JSONL records and converts each example into:

- a prompt
- a completion
- a ground-truth answer

The dataset splits the original gsm8k answer field on `####`, treating the prefix as reasoning text and the suffix as the final answer.

## Prompt Template

`alignment/r1_prompt.py` wraps a template file and generates two things:

1. the prompt shown to the model
2. the supervised response string

The response format is strict:

```text
</think> <answer> ... </answer>
```

That format is not cosmetic. It is part of the evaluation contract.

## Model and Device Setup

`alignment/sft.py`:

- loads a vLLM evaluation model
- builds a device map for the trainable HF model
- loads tokenizer and model weights
- sets `pad_token` if needed
- uses AdamW for optimization

The script is designed for large-model fine-tuning across several GPUs instead of a tiny single-device demo.

## Loss Construction

The key training detail is label masking.

The script tokenizes full prompt-plus-completion sequences, then tokenizes prompts separately to compute prompt lengths. It copies `input_ids` into `labels` and sets prompt positions to `-100`.

That means the model is optimized only on the completion region, not on re-predicting the prompt text.

Padding positions are also masked out with `-100`.

## Gradient Accumulation

The script divides the loss by `gradient_accumulation_steps`, backpropagates on each batch, and only calls `optimizer.step()` after the configured accumulation interval.

That is how the effective batch size can be larger than the physical microbatch size.

## Evaluation

After each epoch, the script loads the policy weights into the vLLM instance and runs `evaluate_math(...)`.

The evaluation path ultimately depends on `r1_zero_reward_fn` in `alignment/drgrpo_grader.py`, which checks both:

- response format compliance
- answer correctness

That is why the SFT example reports two metrics instead of only one accuracy number.

## Why This Example Matters

The repo's homepage proof points come from here:

- zero-shot baseline
- post-SFT accuracy
- format-compliance improvement

More importantly, this chapter shows how the repo handles a realistic instruction-following fine-tuning problem:

- structured prompt templates
- completion-only loss
- evaluation with a format gate
- large-model device placement

So the SFT path is where the project moves from base-model mechanics into a practical downstream task.
