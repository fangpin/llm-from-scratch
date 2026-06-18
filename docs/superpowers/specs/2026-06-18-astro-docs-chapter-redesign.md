# Astro Docs Chapter Redesign

## Summary

Refactor the current `Docs` area from a curated external-link page into a first-class, bilingual documentation section inside the Astro site.

The new docs system should:

1. Keep Markdown as the primary source format.
2. Use English as the default route tree and provide a mirrored Chinese route tree.
3. Organize content by repository module and implementation topic, not by tutorial sequence.
4. Directly absorb and restructure the existing `docs/*.md` files instead of leaving them as a separate parallel corpus.
5. Ground every chapter in the current repository code and results.

This redesign replaces the current v1 constraint of "entry links only" for the Docs page.

## Chosen Direction

Use a chaptered, Markdown-backed docs system with:

1. a docs overview page at `/docs` and `/zh/docs`
2. grouped chapter cards by subsystem
3. dedicated internal chapter pages for each major module
4. mirrored English and Chinese Markdown chapters
5. shared chapter metadata so locale switching works at the chapter level

This is the direct-refactor approach selected by the user.

## Goals

1. Turn `Docs` into a real documentation surface rather than a list of GitHub links.
2. Expand the project introduction into more chapters with deeper implementation detail.
3. Keep the structure module-based:
   - tokenizer
   - transformer core
   - training loop
   - kernel optimization
   - distributed training
   - data pipeline
   - SFT
   - RLFT
4. Make Markdown the durable authoring format for both locales.
5. Reuse and refactor the current `docs/*.md` files where they already contain strong implementation detail.
6. Reduce duplication and retire overlapping material such as `docs/technical_article3.md`.
7. Preserve the site's English-default bilingual model:
   - `/docs/...`
   - `/zh/docs/...`

## Non-Goals

1. Do not rewrite the Python implementation itself.
2. Do not build a full docs product with search, versioning, or blog features in this pass.
3. Do not keep the current flat `docs/*.md` layout unchanged and merely wrap it with routes.
4. Do not make the docs read like a step-by-step tutorial sequence; the primary organizing principle is subsystem architecture.
5. Do not invent claims that are not traceable to the current repository files.

## Current State

### Site

Today the Astro site exposes:

- `/docs`
- `/zh/docs`

but both pages are only external-link grids driven by `site/src/content/site.ts`.

### Existing docs inventory

The repository already contains these Markdown sources:

- `docs/1.md`
- `docs/2.md`
- `docs/3.md`
- `docs/4.md`
- `docs/5-sft.md`
- `docs/qwen25-math-gsm8k-rl-finetune.md`
- `docs/technical_article3.md`

### Source quality assessment

1. `docs/2.md` already provides a deep implementation-oriented explanation of Flash Attention and Triton kernels.
2. `docs/3.md` already provides a deep explanation of DDP and sharded optimizer internals.
3. `docs/4.md` already provides a strong data-pipeline explanation.
4. `docs/5-sft.md` already provides a strong SFT workflow and evaluation explanation.
5. `docs/qwen25-math-gsm8k-rl-finetune.md` already provides a strong RLFT explanation with GRPO and training topology.
6. `docs/1.md` is broad and useful as source material, but it is currently too article-like and should be split into architecture-oriented chapters.
7. `docs/technical_article3.md` substantially overlaps with the distributed-training material and should be merged into the new chapter system instead of surviving as a separate top-level doc.

## Documentation Model

### Information architecture

The docs should be organized into three grouped bands on the overview page.

#### Group 1: Core Stack

1. Project Overview
2. Tokenizer and Vocabulary
3. Transformer Core
4. Training Loop and Checkpointing

#### Group 2: Scale and Performance

5. Flash Attention and Kernel Optimization
6. Distributed Training and Sharded Optimizer
7. Data Processing Pipeline

#### Group 3: Alignment Workflows

8. Supervised Fine-Tuning on gsm8k
9. Reinforcement Learning Fine-Tuning on gsm8k

This is intentionally grouped by subsystem ownership and execution layer, not by a beginner tutorial order.

## Proposed Route Map

### English

- `/docs`
- `/docs/project-overview`
- `/docs/tokenizer`
- `/docs/transformer-core`
- `/docs/training-loop`
- `/docs/flash-attention`
- `/docs/distributed-training`
- `/docs/data-pipeline`
- `/docs/sft-gsm8k`
- `/docs/rlft-gsm8k`

### Chinese

- `/zh/docs`
- `/zh/docs/project-overview`
- `/zh/docs/tokenizer`
- `/zh/docs/transformer-core`
- `/zh/docs/training-loop`
- `/zh/docs/flash-attention`
- `/zh/docs/distributed-training`
- `/zh/docs/data-pipeline`
- `/zh/docs/sft-gsm8k`
- `/zh/docs/rlft-gsm8k`

Slugs stay ASCII and stable across locales so chapter parity is easy to maintain.

## Markdown Source Layout

Refactor the current flat docs directory into a mirrored bilingual tree:

```text
docs/
  en/
    index.md
    project-overview.md
    tokenizer.md
    transformer-core.md
    training-loop.md
    flash-attention.md
    distributed-training.md
    data-pipeline.md
    sft-gsm8k.md
    rlft-gsm8k.md
  zh/
    index.md
    project-overview.md
    tokenizer.md
    transformer-core.md
    training-loop.md
    flash-attention.md
    distributed-training.md
    data-pipeline.md
    sft-gsm8k.md
    rlft-gsm8k.md
```

The new tree should become the canonical site-doc source. The current top-level `docs/*.md` files are inputs to be refactored into this tree, not an extra published surface.

## Chapter Mapping From Existing Files

### Existing source to new chapter mapping

1. `docs/1.md`
   - primary destination: `docs/zh/project-overview.md`
   - secondary material to extract into:
     - `docs/zh/training-loop.md`
     - `docs/zh/tokenizer.md`
     - `docs/zh/transformer-core.md`
   - English peer chapters will be written against the same code paths and repo facts.

2. `docs/2.md`
   - destination: `docs/zh/flash-attention.md`
   - English peer: `docs/en/flash-attention.md`

3. `docs/3.md`
   - primary destination: `docs/zh/distributed-training.md`

4. `docs/technical_article3.md`
   - absorb into `docs/zh/distributed-training.md`
   - remove duplication instead of exposing both docs

5. `docs/4.md`
   - destination: `docs/zh/data-pipeline.md`
   - English peer: `docs/en/data-pipeline.md`

6. `docs/5-sft.md`
   - destination: `docs/zh/sft-gsm8k.md`
   - English peer: `docs/en/sft-gsm8k.md`

7. `docs/qwen25-math-gsm8k-rl-finetune.md`
   - destination: `docs/zh/rlft-gsm8k.md`
   - English peer: `docs/en/rlft-gsm8k.md`

### New chapters that need to be authored from repository code plus README content

The current docs set does not yet provide focused standalone chapters for:

1. tokenizer
2. transformer core
3. training loop and checkpointing

These should be newly authored in both locales using:

- `README.md`
- `README_cn.md`
- `llm/bpe_tokenizer.py`
- `llm/transformer.py`
- `llm/training.py`
- `llm/generating.py`
- `llm/checkpoint.py`

## Chapter Responsibilities

### Project Overview

Purpose:
Explain what the repository implements, how the modules fit together, and where each subsystem lives.

Content responsibilities:

1. repository purpose and teaching orientation
2. high-level module map:
   - `llm/`
   - `kernel/`
   - `parallel/`
   - `data_processing/`
   - `alignment/`
3. training and fine-tuning capability map
4. quick "where to read next" links into the deeper module chapters

### Tokenizer and Vocabulary

Purpose:
Explain how `llm/bpe_tokenizer.py` works from pre-tokenization through merge learning and encode/decode.

Content responsibilities:

1. regex pre-tokenization strategy
2. special-token handling
3. byte-level initialization
4. pair counting and merge ranking
5. encode and decode execution path
6. saved tokenizer artifacts and expected workflow

### Transformer Core

Purpose:
Explain the internals of `llm/transformer.py` as a code-first architecture chapter.

Content responsibilities:

1. custom `Linear` and `Embedding`
2. `RmsNorm`
3. `ScaledDotProductAttention`
4. multi-head attention and causal mask
5. `RoPE`
6. `SwiGlu`
7. block composition and decoder-only structure
8. custom `Softmax` and `CrossEntropyLoss`

### Training Loop and Checkpointing

Purpose:
Explain how model training is actually executed and how the surrounding utilities fit together.

Content responsibilities:

1. `get_batch` data slicing
2. process-group setup and device assignment
3. model initialization
4. training loop, validation loop, logging, and checkpoint cadence
5. cosine learning-rate schedule
6. gradient clipping
7. relation to `llm/generating.py` and `llm/checkpoint.py`

### Flash Attention and Kernel Optimization

Purpose:
Explain the Triton kernel path and how it differs from the reference implementation.

Content responsibilities:

1. why attention becomes a memory bottleneck
2. blockwise Flash Attention strategy
3. Triton kernel structure
4. reference implementation in `flash_attention_mock.py`
5. benchmark scripts and what they compare

### Distributed Training and Sharded Optimizer

Purpose:
Explain the custom distributed path in `parallel/`.

Content responsibilities:

1. DDP parameter broadcast
2. post-accumulate gradient hooks
3. bucket-based synchronization
4. asynchronous all-reduce
5. sharded optimizer partitioning
6. parameter synchronization after step
7. memory tradeoffs versus plain DDP

### Data Processing Pipeline

Purpose:
Explain how raw text is transformed into trainable corpus input.

Content responsibilities:

1. HTML extraction
2. language identification
3. heuristic quality filtering
4. deduplication
5. PII masking
6. harmful-content detection
7. quality classifier
8. how these pieces compose into a pipeline

### Supervised Fine-Tuning on gsm8k

Purpose:
Explain the repo's SFT example as a concrete implementation case, not as a generic alignment essay.

Content responsibilities:

1. dataset preparation
2. R1 prompt template
3. prompt and completion construction
4. label masking and training loss
5. evaluation flow and metrics
6. reported gains and how they are measured

### Reinforcement Learning Fine-Tuning on gsm8k

Purpose:
Explain the RLFT implementation in `alignment/` with emphasis on reward design and execution topology.

Content responsibilities:

1. reward function and format gating
2. grouped rewards and normalized advantages
3. REINFORCE, baseline, and GRPO distinctions
4. response masking
5. policy/reference/sample/eval device split
6. training and evaluation flow

## Overview Page Design

The docs overview page should stop looking like an external link hub.

It should contain:

1. a short docs intro describing the docs as a system map of the repository
2. grouped chapter sections by subsystem band
3. chapter cards with:
   - title
   - one-sentence summary
   - key source files
   - implementation focus
4. a visual system-map strip near the top that shows:
   - tokenizer -> model -> training -> scale -> alignment
5. internal links only for the main chapter cards

The overview page should optimize for scanning and selection, not for reading everything on one long page.

## Chapter Page Design

Each chapter page should be a Markdown-driven content page wrapped in the existing site shell.

Recommended chapter-page structure:

1. title
2. short summary
3. metadata row:
   - chapter group
   - key code paths
   - related chapter links
4. rendered Markdown body
5. bottom navigation to previous and next sibling chapters

The visual treatment should remain consistent with the current site:

- dark technical theme
- restrained accents
- readable long-form layout
- no decorative card nesting

## Markdown Content Requirements

Every chapter Markdown file should carry frontmatter similar to:

```yaml
title: Transformer Core
summary: Decoder-only transformer internals implemented in llm/transformer.py.
slug: transformer-core
locale: en
group: core-stack
order: 3
translationKey: transformer-core
sourceFiles:
  - llm/transformer.py
sourceDocs:
  - docs/1.md
```

The frontmatter should be sufficient to drive:

1. overview page grouping
2. chapter order
3. locale switching
4. related-file display

## Astro Integration Strategy

### Recommended content-loading model

1. Add a content source for the new docs tree rather than storing long-form docs inside `site/src/content/site.ts`.
2. Prefer loading repo-level Markdown from `docs/en/*.md` and `docs/zh/*.md` so Markdown remains the single source of truth.
3. Add a dynamic docs route for chapter pages instead of hand-authoring one `.astro` file per chapter.

### Expected site changes

Likely touched areas:

1. `site/src/content/site.ts`
   - reduce docs page from card literals to overview copy only
2. `site/src/lib/locale.ts`
   - expand locale-switch logic from fixed docs root to chapter-level docs routes
3. `site/src/pages/docs/index.astro`
   - switch from external cards to internal chapter overview
4. `site/src/pages/zh/docs/index.astro`
   - same for Chinese
5. add content config and docs page templates under `site/src/`
6. add docs-specific tests for route parity and chapter metadata

## Markdown Rendering Requirements

The current docs already contain:

1. code fences
2. tables
3. Mermaid diagrams

The site-side markdown pipeline must preserve these well enough for the docs not to regress in readability.

In particular, the implementation should explicitly account for Mermaid support because several existing docs use Mermaid flowcharts.

## Locale Strategy

1. English remains the default site locale.
2. Chinese remains a first-class mirrored locale under `/zh/`.
3. Locale switching should work per chapter:
   - `/docs/distributed-training` <-> `/zh/docs/distributed-training`
4. English and Chinese chapters should be authored as peer Markdown files, not runtime-translated strings.
5. Shared slugs plus a `translationKey` should be used to map peer chapters.

## Migration Decisions

1. The old flat `docs/*.md` files are inputs to be restructured.
2. The new bilingual chapter tree becomes the published docs structure.
3. Duplicate distributed-training material from `docs/technical_article3.md` is absorbed into the new distributed-training chapter.
4. The old docs overview page in the Astro site is replaced, not extended with another parallel docs surface.

## Acceptance Criteria

1. `/docs` and `/zh/docs` become real overview pages with grouped internal chapter cards.
2. Every chapter exists in both locales.
3. English is still the default locale.
4. Chapter pages are Markdown-backed rather than hardcoded in `site/src/content/site.ts`.
5. Existing high-value material from `docs/2.md`, `docs/3.md`, `docs/4.md`, `docs/5-sft.md`, and `docs/qwen25-math-gsm8k-rl-finetune.md` is preserved through refactoring.
6. The new docs structure reduces duplication instead of adding more overlapping documents.
7. Locale switching works from chapter page to chapter page.
8. The current docs page no longer opens the main chapter cards in external GitHub tabs.

## Risks and Mitigations

### Risk: markdown migration grows too large

Mitigation:
Keep the first implementation to the nine planned chapters and avoid adding secondary appendices.

### Risk: bilingual upkeep becomes expensive

Mitigation:
Use a shared frontmatter schema and identical slugs across locales so parity is visible and testable.

### Risk: Mermaid diagrams regress when moved into the site

Mitigation:
Treat Mermaid rendering as an explicit implementation requirement rather than an optional polish item.

### Risk: current docs are uneven in tone

Mitigation:
Rewrite chapter intros and summaries into a consistent technical style while preserving the implementation details that are already strong.

## Review Focus

Before implementation, confirm these decisions:

1. the nine-chapter module structure
2. the English and Chinese mirrored Markdown tree under `docs/en` and `docs/zh`
3. merging `docs/technical_article3.md` into the distributed-training chapter
4. using internal docs routes instead of GitHub outbound links for the primary docs experience
