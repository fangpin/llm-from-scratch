---
title: Tokenizer and Vocabulary
summary: How the repository trains and applies a byte-pair tokenizer, from regex pre-tokenization to merge learning and byte-level encode/decode.
slug: tokenizer
locale: en
group: core-stack
order: 2
translationKey: tokenizer
sourceFiles:
  - llm/bpe_tokenizer.py
  - llm/args.py
sourceDocs:
  - docs/1.md
---

# Tokenizer and Vocabulary

`llm/bpe_tokenizer.py` implements a byte-pair tokenizer from scratch. The file owns both the offline training phase and the online encode/decode phase, so it is the first true systems boundary in the repo.

## Internal State and Invariants

The tokenizer maintains four core data structures:

- `vcab2id: dict[bytes, int]`
- `id2vcab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `merge_ranks: dict[tuple[bytes, bytes], int]`

`vcab2id` and `id2vcab` are the runtime vocabulary. `merges` records the BPE merge history in training order. `merge_ranks` is derived from `merges` and is what inference actually uses when it greedily decides which adjacent pair to collapse next.

The implementation uses `bytes` as the canonical token representation. That matters because:

- the initial vocabulary is the full 256 byte range
- merge rules operate on byte fragments
- decoding is just byte concatenation followed by UTF-8 decode

So the tokenizer never needs a separate "subword object" abstraction.

## Regex Pre-tokenization

Before merge logic runs, the code segments text with:

```python
self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
```

This pre-tokenizer groups together:

- contractions
- letter runs
- number runs
- punctuation runs
- whitespace spans

That means BPE is learned inside already segmented spans, not over one giant stream of raw Unicode characters. In practice, this reduces pathological merges across word and punctuation boundaries while still leaving byte-level flexibility inside each pre-token.

## Special Tokens

When `special_tokens` are supplied, `_pre_token()` first splits the byte stream around those exact byte sequences. Anything that matches a special token bypasses the regex tokenization pass and is inserted as an atomic token.

This is why `<|endoftext|>` can be preserved across:

- tokenizer training
- corpus encoding
- generation stop conditions

Without that branch, a special token would be broken into bytes and then possibly merged inconsistently.

## Training Phase

`train()` performs the full BPE construction.

### Step 1: Initialize the base vocabulary

The method first installs:

1. special-token ids
2. all 256 raw byte ids

That gives the tokenizer a complete fallback vocabulary before any learned merges exist.

### Step 2: Build token-count statistics

The input corpus is read as UTF-8 bytes, then `_pre_token()` is applied. For each normal pre-token, the code creates a tuple of one-byte symbols:

```python
bs = tuple(bytes([b]) for b in pre_token)
```

The dictionary `tokens_cnt` counts how many times each byte tuple appears. This is effectively a word-frequency table, except "word" here means "regex pre-token represented as bytes."

### Step 3: Count adjacent pairs

The code then scans every token tuple and accumulates counts of adjacent pairs into `pair_cnt`. This is the sufficient statistic BPE needs to choose the next merge.

### Step 4: Merge loop

While the current vocabulary is smaller than `vocab_size`:

1. pick the most frequent pair from `pair_cnt`
2. append it to `self.merges`
3. create a new vocabulary entry for the concatenated bytes
4. rewrite every affected token tuple
5. decrement counts for old pairs
6. increment counts for the new pairs

This implementation uses the helper `update_pair_counts()` to keep pair statistics consistent incrementally, rather than recomputing all pair counts from scratch after each merge.

That choice is the main algorithmic optimization in the file.

### Step 5: Finalize lookup maps

At the end, `id2vcab` and `merge_ranks` are rebuilt from the final state. `merge_ranks` maps each merge pair to its training-time priority:

```python
self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
```

Inference then replays these ranks greedily.

## Encoding Phase

`encode()` applies the trained tokenizer to text.

### Step 1: Convert to bytes and pre-tokenize

The text is UTF-8 encoded and split by `_pre_token()`. Special tokens are detected first and converted directly to ids.

### Step 2: Expand each pre-token into byte symbols

For normal pre-tokens, the code starts with:

```python
tokens = tuple(bytes([c]) for c in pre_token)
```

So every token begins as raw bytes again, regardless of how many merges were learned during training.

### Step 3: Greedy merge replay

The loop repeatedly:

1. enumerates adjacent pairs
2. finds the pair with the lowest merge rank
3. merges that one pair
4. repeats until no ranked pair remains

This is the exact "BPE as learned merge replay" mechanism. Training learns a global ordering; inference only consults that ordering and never recomputes pair frequencies.

The behavior is greedy and deterministic. There is no beam search or sampling in tokenization.

## Decoding Phase

`decode()` reverses the process by:

1. looking up each token id in `id2vcab`
2. concatenating the corresponding bytes
3. decoding the byte stream as UTF-8

If an id is missing, the code substitutes the replacement-character bytes:

```python
b"\xef\xbf\xbd"
```

That keeps decode robust in the face of corrupted inputs or mismatched checkpoints.

## Persistence and Reuse

`save()` serializes:

- `merge`
- `id2vcab`
- `special_tokens`

The file intentionally does not serialize every derived field. `load()` calls `from_pretrained()` and reconstructs `vcab2id` plus `merge_ranks`. That keeps the saved artifact minimal while still restoring the full runtime contract.

## Corpus Preparation Path

The `__main__` block shows how the tokenizer is used operationally:

1. parse CLI args from `llm.args`
2. train tokenizer on the source corpus
3. save tokenizer checkpoint
4. encode training text
5. persist token ids to `.npy`
6. encode validation text
7. persist validation token ids

So the tokenizer is not a side demo. It is the producer for the pretraining dataset consumed by `llm/training.py`.

## Design Tradeoffs

This implementation is compact, but a few tradeoffs are worth noting:

- It reads the training corpus into memory before pre-tokenization.
- Encoding searches all adjacent pairs linearly at each merge step.
- The vocabulary field is spelled `vcab`, and docs should follow the code rather than silently renaming it.

Those tradeoffs are acceptable for the repo's goal: clarity first, scale second. The important point is that every stage of BPE is explicit and inspectable.
