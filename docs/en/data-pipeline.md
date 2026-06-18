---
title: Data Processing Pipeline
summary: The preprocessing stack in data_processing/, from HTML extraction and language filtering to deduplication, PII masking, harmful-content checks, and quality classification.
slug: data-pipeline
locale: en
group: scale-performance
order: 7
translationKey: data-pipeline
sourceFiles:
  - data_processing/html_process.py
  - data_processing/language_identification.py
  - data_processing/quality_filter.py
  - data_processing/deduplicate.py
  - data_processing/mask_pii.py
  - data_processing/harmful_detect.py
  - data_processing/quality_classfier.py
sourceDocs:
  - docs/4.md
---

# Data Processing Pipeline

The repo includes a preprocessing stack in `data_processing/` for turning noisy raw corpora into cleaner training input.

The files are independent utilities, but together they form a pipeline:

1. extract text
2. keep the right language
3. remove obviously weak documents
4. remove duplicates
5. mask sensitive data
6. filter harmful content
7. classify quality more precisely

## HTML Extraction

`html_process.py` uses `resiliparse` for two tasks:

- encoding detection
- plain-text extraction from HTML

That matters because web-scale corpora are often scraped as raw HTML with mixed encodings and a large amount of structural noise.

## Language Identification

`language_identification.py` loads a FastText language-identification model and predicts labels such as `en` or `zh`.

The file strips the `__label__` prefix from FastText outputs and returns both labels and probabilities.

This is the language gate that can narrow a crawl down to the target training language.

## Heuristic Quality Filters

`quality_filter.py` applies simple rules such as:

- word count range
- average word length
- alphabetic-token proportion
- ellipsis-heavy line ratio

These rules are not meant to be perfect. They are designed to cheaply remove obviously low-value documents before more expensive or more nuanced filtering happens.

## Deduplication

`deduplicate.py` implements two levels of deduplication:

- exact line deduplication
- approximate document deduplication with MinHash-style signatures

The approximate path normalizes text, tokenizes it, creates n-gram shingles, builds multiple hash signatures, and uses banded signature buckets to find candidate duplicates.

That is a practical compromise between "only remove exact duplicates" and "compute full pairwise similarity for every document."

## PII Masking

`mask_pii.py` uses regexes to detect and replace:

- email addresses
- phone numbers
- IPv4 addresses

The replacement tokens are explicit placeholders such as `|||EMAIL_ADDRESS|||`.

That makes it possible to sanitize documents without discarding them entirely.

## Harmful-Content Detection

`harmful_detect.py` defines:

- `NSFWDetector`
- `ToxicDetector`

Both are FastText-backed classifiers that return labels and probabilities after newline normalization.

This gives the data pipeline a basic safety filter before documents reach training.

## Quality Classification

The repo also includes `quality_classfier.py`, a FastText-based classifier meant to separate higher-quality text from lower-quality text.

This layer sits above heuristic filters. The heuristics remove obvious noise; the classifier is where a project-specific notion of quality can become more nuanced.

## Why the Pipeline Matters

The core model and training scripts are only as good as the text they consume. This preprocessing stack is the bridge between noisy external corpora and a trainable token-id dataset.

That is why the repo keeps data-processing code in the same implementation surface as model code. It treats data preparation as part of the system, not as an external afterthought.
