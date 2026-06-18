---
title: Reinforcement Learning Fine-Tuning on gsm8k
summary: How the repo implements reward shaping, grouped normalization, GRPO-style clipping, and multi-GPU role separation for RLFT.
slug: rlft-gsm8k
locale: en
group: alignment-workflows
order: 9
translationKey: rlft-gsm8k
sourceFiles:
  - alignment/train_rl.py
  - alignment/grpo.py
  - alignment/drgrpo_grader.py
  - alignment/evaluate.py
  - alignment/args.py
sourceDocs:
  - docs/qwen25-math-gsm8k-rl-finetune.md
---

# Reinforcement Learning Fine-Tuning on gsm8k

The RLFT path extends the same gsm8k problem used in SFT, but moves into policy-gradient training with explicit reward design and a more complex device topology.

The main pieces are:

- `alignment/train_rl.py`
- `alignment/grpo.py`
- `alignment/drgrpo_grader.py`

## Reward Function

The reward function is `r1_zero_reward_fn`.

It does not only ask whether the final numeric answer is right. It first checks whether the model respected the required structure:

- `</think> <answer>`
- `</answer>`

Only properly formatted outputs can receive a full reward.

That makes format compliance part of the policy objective rather than just a reporting metric.

## Grouped Reward Normalization

`compute_group_normalized_rewards()` computes rewards for multiple responses sampled from the same prompt, then normalizes them inside each group.

The function:

1. computes raw rewards
2. reshapes them into `[n_prompts, group_size]`
3. subtracts the group mean
4. optionally divides by group standard deviation

This creates relative advantages rather than using raw rewards directly for every policy update.

## Loss Variants

`compute_policy_gradient_loss()` exposes three modes:

- `no_baseline`
- `reinforce_with_baseline`
- `grpo_clip`

For `grpo_clip`, the implementation compares new-policy log probabilities with detached old-policy log probabilities, computes an importance ratio, and clamps that ratio inside a configured clip range.

That gives the RL path a PPO-like stability mechanism while staying compact enough to read.

## Response-Only Masking

`train_rl.py` builds a `response_mask` per microbatch so that loss averaging only applies to the generated response region, not the prompt prefix or padding positions.

That keeps optimization focused on what the policy actually chose to generate.

## Device Topology

One of the most distinctive features of this RLFT path is how it splits work across GPUs:

- a vLLM sampling model
- a frozen reference model
- an evaluation model
- a trainable policy model distributed across the remaining GPUs

`partition_model_across_devices()` manually assigns layers to GPUs instead of relying on a fully opaque auto-placement strategy.

That makes the training topology inspectable and controllable.

## Training Loop Shape

The RL training loop does the following:

1. sample grouped prompts
2. generate rollouts with vLLM
3. compute rewards and normalized advantages
4. tokenize prompt-plus-response text
5. compute current-policy and reference-policy log probabilities
6. build response masks
7. run microbatch policy-gradient updates
8. clip gradients and step the optimizer

Periodic checkpoints are evaluated with the math evaluator just like the SFT path.

## Why This Example Matters

This is the strongest example in the repo of moving beyond "from-scratch pretraining" into a full downstream optimization workflow:

- reward design
- grouped preference-style normalization
- clipped policy updates
- multi-role multi-GPU execution

That makes the RLFT chapter the best place to understand how the repository handles training systems work once the model is already useful and the optimization target becomes behavior rather than pure next-token likelihood.
