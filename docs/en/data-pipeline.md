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

`data_processing/` is a toolbox for turning messy raw corpora into something the tokenizer and training loop can consume. The repository does not ship one monolithic preprocessing runner. Instead it exposes a set of small utilities that can be composed into a pipeline.

## Pipeline Ordering

A practical execution order for these files is:

1. `html_process.py`
2. `language_identification.py`
3. `quality_filter.py`
4. `deduplicate.py`
5. `mask_pii.py`
6. `harmful_detect.py`
7. `quality_classfier.py`

That order reflects cost and selectivity:

- structural cleanup first
- cheap deterministic filters next
- heavier deduplication and classifier stages later

The codebase keeps those stages separate so you can insert or skip them depending on the corpus.

## `html_process.py`: HTML to Plain Text

`extract_text_from_html(html_content: bytes)` is deliberately small. It:

1. detects encoding with `resiliparse.parse.encoding.detect_encoding`
2. decodes raw bytes into a Python string
3. extracts visible text with `resiliparse.extract.html2text.extract_plain_text`

So the file's contract is not "parse a crawl dataset end to end." Its contract is narrower: given HTML bytes, return plain text robustly across mixed encodings.

The file also imports `ArchiveIterator` and `WarcRecordType`, which suggests the intended upstream caller may be WARC/Common Crawl style code, but the helper itself stays at raw-HTML granularity.

## `language_identification.py`: FastText Wrapper Plus Compatibility Patch

This file does two things at once.

First, it monkey-patches `fasttext.FastText._FastText.predict` to avoid a NumPy 2 compatibility problem around array creation. That patch is applied at import time, before any model is loaded.

Second, it defines `LanguageIdentifier`, which:

- loads `pre_trained/lid.176.bin`
- replaces newlines and carriage returns with spaces
- calls `model.predict`
- strips the `__label__` prefix from FastText labels

`identify(text, k=1)` returns:

- `(label, prob)` when `k == 1`
- `(labels, probs)` when `k > 1`

So the file is both a model wrapper and a runtime compatibility shim.

## `quality_filter.py`: Cheap Heuristic Filters

`QualityFilter` is the deterministic screening stage. Its constructor downloads the NLTK resources it needs:

- `punkt`
- `punkt_tab`

It then exposes four filters:

- `pass_wc_filter`: keep texts with 50 to 100000 tokens
- `pass_word_len_filter`: keep texts whose average token length is between 3 and 10
- `pass_alphabetic_filter`: require at least 80 percent of tokens to contain an alphabetic character
- `pass_ellipsis_filter`: require fewer than 30 percent of lines to end with `...`

`pass_all_filters(content)` tokenizes once with `word_tokenize` and then ANDs those checks together.

These are not semantic quality judgments. They are cheap guards that remove obvious noise before slower steps run.

## `deduplicate.py`: Exact and Approximate Deduplication

This file is the most algorithmically dense module in the preprocessing stack.

### Exact line deduplication

`exact_line_deduplicate(files, output_dir)`:

1. reads every line from every file
2. counts `hash(line)` occurrences
3. rewrites only the unique lines to an output file with the same basename

This is a simple structural deduplicator for repeated boilerplate, headers, and templated lines.

### `MinHashDeduplicator`

The approximate dedup path is built around `MinHashDeduplicator`. Its constructor configures:

- `num_hashes`
- `num_bands`
- `n_gram`
- `jaccard_threshold`

The approximate pipeline is:

1. normalize text
2. tokenize
3. build n-gram shingles
4. compute a MinHash signature with multiple seeded SHA256 hashers
5. place documents into candidate buckets
6. verify candidate pairs with exact Jaccard similarity
7. copy only survivor files to the output directory

### Normalization and shingles

`normalize()` does more than lowercase:

- Unicode NFD normalization
- accent stripping through `unidecode`
- lowercase conversion
- punctuation removal except alphanumerics and spaces
- whitespace collapsing

`shingle()` then tokenizes the normalized text and produces contiguous token n-grams as a set.

One implementation detail worth noticing is that `_tokenize()` downloads `punkt` and `punkt_tab` inside the helper itself, so NLTK setup can happen repeatedly unless the resources are already cached.

### Signatures and buckets

`signatures()` computes one MinHash value per seeded hasher by taking the minimum hash across all shingles.

The bucket construction in `deduplicate()` is a sliding-window style scheme:

```python
for i in range(self.num_hashes - self.num_bands):
    bucket_id = (i, tuple(signatures[i : i + self.num_bands]))
```

So candidate buckets are built from overlapping slices of the signature vector rather than from one textbook fixed partition of bands and rows.

### Candidate verification and survivors

Once candidate pairs are assembled, the file computes exact Jaccard similarity on the normalized shingle sets. If the similarity exceeds `jaccard_threshold`, one file is marked as duplicate.

The current implementation keeps the second filename in the sorted pair and adds only the first one to `deduplicates`. That makes survivor selection deterministic but order-dependent.

## `mask_pii.py`: Regex-Based Sanitization

`PIIMasker` compiles regular expressions for:

- email addresses
- phone numbers
- IPv4 addresses

Each masking method returns a tuple:

```python
(masked_content, count)
```

and replaces matches with explicit placeholders such as:

- `|||EMAIL_ADDRESS|||`
- `|||PHONE_NUMBER|||`
- `|||IP_ADDRESS|||`

That design keeps the document usable for training while removing the sensitive token surface.

## `harmful_detect.py`: Safety Classifiers

This file defines two FastText-backed wrappers:

- `NSFWDetector`
- `ToxicDetector`

Both classes:

- load a pre-trained binary at initialization
- normalize newlines to spaces before inference
- strip the `__label__` prefix from outputs

So the harmful-content stage mirrors the language-ID stage: thin wrapper, consistent label cleanup, model file decided by the constructor.

## `quality_classfier.py`: Learned Quality Scoring

The final stage is model-based quality classification.

The file contains both:

- `train(...)`, which builds a FastText supervised classifier from labeled text and saves it to `checkpoints/quality_classifier.bin`
- `QualityClassifier.identify(...)`, which loads the saved model and returns a normalized label/confidence pair

`identify()` also handles a few edge cases explicitly:

- blank strings return `("low-quality", 0.0)`
- unknown labels are mapped back to `("low-quality", 0.0)`
- labels are normalized from `high_quality` to `high-quality`

The repository path is intentionally spelled `quality_classfier.py`, and the docs keep that spelling to match the real file.

## How It Connects to the Rest of the Repository

This module boundary exists before tokenization:

- cleaned text flows into `llm/bpe_tokenizer.py`
- deduplication and quality filtering raise information density before token IDs are created
- PII and harmful-content filters constrain what the model is allowed to learn from

Keeping preprocessing outside `llm/` means you can change corpus policy without changing tokenizer or model code.

## Design Tradeoffs

The current preprocessing stack is explicit rather than fully productized:

- there is no single orchestrator script
- some NLTK resources are downloaded at runtime
- exact dedup is hash-based and file-oriented
- approximate dedup uses a practical bucket heuristic rather than a full platform service
- model-backed filters depend on external FastText binaries

That fits the repository's goal. The module is showing the building blocks of a real preprocessing pipeline, not hiding them behind one opaque command.
