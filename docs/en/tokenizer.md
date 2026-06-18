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

The tokenizer is implemented in `llm/bpe_tokenizer.py` as a from-scratch byte-pair encoding pipeline. It is not a thin wrapper over a third-party tokenizer library. The file owns the full path from raw text to token ids.

## Design Shape

The tokenizer keeps four main pieces of state:

- `vcab2id`: bytes to integer ids
- `id2vcab`: reverse mapping for decode
- `merges`: learned byte-pair merges in training order
- `merge_ranks`: a rank map used during greedy encoding

The implementation uses byte strings as the canonical vocabulary representation. That matters because BPE operates on byte-level fragments before those fragments become larger merged symbols.

## Pre-tokenization

The tokenizer starts with a regex pattern:

```python
self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
```

This breaks text into coarse pieces before merge logic runs. The pattern separates:

- letter sequences
- number sequences
- punctuation fragments
- whitespace segments

That means merge learning does not start from raw Unicode code points across the whole file. It starts from byte-level pieces inside these regex-derived pre-tokens.

## Special Tokens

If `special_tokens` are provided, `_pre_token()` splits the byte stream around those tokens first. Each special token is preserved as an atomic unit and bypasses the normal regex tokenization logic.

This is how the repo keeps separators like `<|endoftext|>` stable across training, encoding, and generation.

## Training Path

The training loop in `train()` follows a standard but explicit BPE recipe:

1. initialize ids for special tokens
2. initialize ids for all 256 raw byte values
3. pre-tokenize the corpus
4. count word-level byte tuples
5. count adjacent byte-pair frequencies
6. repeatedly merge the most frequent pair until the target vocab size is reached

The core merge update is local and mechanical:

- pick the best pair from `pair_cnt`
- create a new merged symbol
- rewrite matching words
- decrement old pair counts and increment new pair counts

The file does not hide any of this behind a trainer abstraction, which makes it useful as a teaching implementation.

## Encoding

`encode()` converts text to bytes, pre-tokenizes it, then handles each pre-token separately.

For a normal pre-token:

1. split it into single-byte symbols
2. repeatedly scan adjacent pairs
3. choose the pair with the lowest merge rank
4. merge that pair
5. continue until no ranked pair remains

The key detail is that encoding is greedy over learned merge ranks, not over fresh pair counts. Training decides the merge list; inference replays that learned ordering.

## Decoding

`decode()` reverses the process by looking up each id in `id2vcab`, concatenating bytes, and decoding the byte string back to UTF-8.

Unknown ids fall back to the replacement character bytes:

```python
b"\xef\xbf\xbd"
```

That makes decode robust even if the input ids are not perfectly aligned with the saved vocabulary.

## Persistence

The tokenizer stores three durable artifacts in `save()`:

- merges
- `id2vcab`
- special tokens

`load()` restores those objects and rebuilds the derived maps.

That structure is enough for training data preparation, checkpoint-time reuse, and generation-time reuse. The generator loads the tokenizer from `args.tokenizer_checkpoint` before it samples text from a model checkpoint.

## Operational Role in the Repo

The tokenizer is not an isolated demo. It feeds:

- `llm.training` for corpus-to-token-id preparation
- `llm.generating` for prompt encoding and completion decoding
- any downstream workflow that expects the repo's checkpoint format

In other words, the tokenizer is the start of the whole model pipeline, not a side utility.
