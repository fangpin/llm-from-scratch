---
title: Docs
summary: Module-based documentation for the repository, organized around implementation layers instead of tutorial order.
slug: index
locale: en
group: core-stack
order: 0
translationKey: docs-index
sourceFiles:
  - README.md
  - README_cn.md
sourceDocs:
  - docs/1.md
---

This docs section is a system map for the repository rather than a beginner tutorial.

Each chapter is written from the code outward:

- ownership boundaries between modules
- the concrete control flow through functions and classes
- the tensors or records that move across those boundaries
- the current implementation limits that matter when you run or extend the code

The chapters are grouped into three layers:

- the core stack that turns text into a decoder-only model
- the scale-and-performance layer that covers kernels, distributed execution, and preprocessing
- the alignment workflows that fine-tune and evaluate a math model on gsm8k

Two reading paths work well:

- start with `Project Overview` if you want the whole repository topology first
- start with `Tokenizer and Vocabulary` if you want to follow the actual data path from raw text to training batches

Use the overview cards below to jump directly to the subsystem you want to inspect.
