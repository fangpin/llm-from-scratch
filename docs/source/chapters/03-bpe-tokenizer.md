# BPE tokenizer

```{contents} 本页目录
---
depth: 2
local: true
---
```

训练 tokenizer 前，先要解决一个更底层的问题：输入文本到底是什么。Unicode 把字符映射到 code point，例如 `牛` 对应一个整数 code point；但直接在 code point 上建词表会很稀疏，因为 Unicode 覆盖的字符非常多，而且不同语言、符号和控制字符的频率差异巨大。真正更稳的入口是 UTF-8 bytes：任意 Unicode 字符串都可以编码成 0 到 255 的 byte 序列。

这个选择带来一个很强的不变量：只要我以 byte 为初始词表，就不会有 out-of-vocabulary。初始 vocabulary 只有 256 个 byte token，加上特殊 token。任何输入文本，即使包含罕见字符、emoji 或混合语言，也一定可以先表示成 byte 序列。代价是序列会变长。比如英文字符通常是一个 byte，日文、中文字符在 UTF-8 里通常是多个 byte；如果直接按 byte 建模，模型会被迫在很长的低级序列上学习语言规律。

BPE 的位置就在这里：它保留 byte-level 的覆盖能力，同时通过 merge 高频相邻 byte 序列来压缩长度。可以把它看成一个只会做局部替换的压缩算法：如果 `b"the"` 经常出现，我就希望它从三个 byte token 变成一个 token；如果 `b"ing"` 常出现，也可以逐步变成一个 token。词表变大了，但序列变短了，模型训练通常更划算。

BPE（Byte Pair Encoding，字节对编码）是一种**子词（subword）分词算法**，广泛应用于现代大语言模型（如 GPT、LLaMA、BERT 等）的 tokenizer 中。它的核心思想是：**在词（word）和字符（character）之间找到一个平衡点，既能处理未登录词（OOV），又能控制词汇表大小**。

![](../assets/images/03-bpe-tokenizer/image-01.png)

## 为什么需要 BPE？——传统分词的痛点


| 分词方式 | 问题 |
| --- | --- |
| 按词分词（Word-level） | 词汇表巨大（百万级），无法处理未见过的词（如 "unhappiness" 拆成 "un", "happy", "ness" 更合理） |
| 按字符分词（Char-level） | 序列太长（"hello" → ['h','e','l','l','o']），模型难以捕捉语义单元 |


→ **BPE 提出“子词”粒度**：将词拆分为高频出现的子词单元（如 "playing" → "play" + "ing"）。

## BPE 的核心原理（两阶段）

### 阶段 1️⃣：**构建词汇表（训练阶段）**

1. **初始化**：

   - 将所有单词拆成字符，并加上结束符（如 `</w>` 表示词尾）  
   例：`["low</w>", "lower</w>", "newest</w>", "widest</w>"]`
2. **迭代合并最高频的相邻字节对**：

   - 统计所有相邻符号对的频率
   - 合并频率最高的 pair，加入词汇表
   - 重复直到达到预设词汇表大小（如 30,000）

#### 📌 示例（简化版）：

初始：  
`l o w </w>`  
`l o w e r </w>`  
`n e w e s t </w>`  

- 最高频 pair: `('l', 'o')` → 合并为 `lo`  
- 新序列：`lo w </w>`, `lo w e r </w>`, ...  
- 下一轮：`('lo', 'w')` → `low`  
- 再下一轮：`('e', 'r')` → `er`，`('e', 's')` → `es`，等等

最终词汇表包含：`l, o, w, e, r, n, s, t, </w>, lo, low, er, es, est, ...`

---

### 阶段 2️⃣：**分词（推理阶段）**

给定一个新词（如 `"lowest"`），用贪心最长匹配从词汇表中拆分：

1. 初始化：`l o w e s t </w>`
2. 查找最长可匹配子词：

   - `low` ∈ vocab → 合并 → `low e s t </w>`
   - `es` ∈ vocab → `low es t </w>`
   - `est` ∈ vocab（如果存在）→ `low est </w>`
3. 最终结果：`["low", "est"]`（或 `["low", "es", "t"]`，取决于 vocab）

> ✅ **关键特性**：即使 `"lowest"` 未在训练集中出现，也能被合理拆分！

---

## BPE 在 LLM 中的实际实现细节

虽然叫 “Byte Pair Encoding”，但现代实现（如 Hugging Face `tokenizers`）通常基于 **Unicode 字符**而非原始字节，更准确应称为 **Subword Regularization with BPE**。

### 常见变体与增强：


| 技术 | 说明 |
| --- | --- |
| Byte-level BPE（如 GPT-2/3） | 先将文本转为 UTF-8 字节（256 个 token），再在其上训练 BPE<br>✅ 优势：能处理任意 Unicode 字符（包括 emoji、中文、罕见符号） |
| Pre-tokenization | 先用规则分割（如按空格、标点），再对每个片段做 BPE<br>例："Hello, world!" → ["Hello", ",", "world", "!"] → 再分词 |


- 预分词：先按语义分割一遍输入，再在子词上做分词，防止将跨语言的token 合并到一起。GPT2 中被提出，后来被广为使用的预分词方式是：基于正则表达式的预分词——r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?\[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""，它使得：

  - 单词、数字、标点、空格等被合理分离；
  - 保留空格信息（用特殊符号如 `Ġ` 表示）；



## BPE 的优缺点

### ✅ 优点：

- **解决 OOV 问题**：任何词都能被拆分为子词
- **词汇表可控**：通常 10k–50k，远小于 word-level
- **语言无关**：适用于中、英、德等所有语言（尤其适合黏着语如德语、土耳其语）
- **高效**：分词速度快，适合线上服务

### ❌ 缺点：

- **语义可能割裂**：如 `"unbelievable"` 拆成 `"un", "bel", "iev", "able"`（若 vocab 不够大）
- **同一词多种分法**：受 pre-tokenization 和 vocab 影响
- **中文效果一般**：中文天然以字为单位，BPE 优势不如英文明显（但 byte-level BPE 仍可用）



## BPE 实现

### 实现思路

实现 BPE 时，最容易写错的是边界。算法表面上很简单：统计所有相邻 token pair 的频率，选择最高频 pair，把它合并成一个新 token，然后重复直到词表达到目标大小。但这里的“相邻”不是在整个文件上无条件相邻。

第一层边界来自特殊 token。像 `<|endoftext|>` 这种字符串通常表示文档分隔符，它应该作为一个独立 token 被保留。训练 merge 统计时，先按特殊 token 切开语料，让它们成为硬边界。这样做的原因很具体：如果两个文档之间恰好出现 `...word<|endoftext|>next...`，我们不希望 tokenizer 学到一个跨文档边界的 token。特殊 token 本身也不应该贡献普通 merge 统计，否则它会污染正常文本的压缩规则。

第二层边界来自 pre-tokenization。现代 GPT 系列 tokenizer 常用 regex 把文本粗分成类似单词、数字、标点、空白的小段，然后只在每个 pre-token 内部做 BPE merge。这样可以避免把语义上不该绑定的片段过早粘在一起。例如 `dog!` 和 `dog.` 不应该因为标点不同就完全变成两套互不共享的 token 路径；pre-tokenizer 让 merge 主要发生在更稳定的局部片段中。

```text
vocab = 256 byte tokens + special tokens
pretoken_counts: dict[tuple[bytes, ...], int]
merges: list[tuple[bytes, bytes]]

repeat until len(vocab) == vocab_size:
    pair_counts = count_adjacent_pairs(pretoken_counts)
    best_pair = highest_count_pair(pair_counts)
    break ties by lexicographically greater pair
    replace best_pair inside each pre-token
    append best_pair to merges and add concatenated bytes to vocab
```

上面这个朴素版本足够说明语义，但不够快。慢的根源是每次 merge 后都全量重扫所有 pre-token。真正值得优化的是 pair count 的增量更新：一次 merge 只会影响和被合并 pair 相邻的 pair，因此我可以维护 pair 到出现位置或受影响 pre-token 的索引，只更新局部计数。pre-tokenization 本身也常是瓶颈，因为它要扫完整语料；这里可以用 multiprocessing 按文档边界分块并行。注意这两类优化的性质不同：pre-tokenization 容易并行，merge 过程因为每一步依赖上一步结果，在 Python 里很难直接并行。

### BPE 编码和解码：训练时学到的顺序必须完整复用

训练 BPE 得到的是两个持久化对象：`vocab: dict[int, bytes]` 和 `merges: list[tuple[bytes, bytes]]`。编码新文本时，我们不能重新统计频率，也不能贪心选择当前最长匹配来替代训练过程。正确做法是先用同一个 pre-tokenizer 切分文本，把每个 pre-token 转成 UTF-8 byte token 序列，然后按训练时创建 merge 的顺序依次应用可用 merge。

这个“按顺序应用”很关键。BPE merge 有历史依赖：先 merge `(b"t", b"h")`，后面才可能出现 `(b"th", b"e")`。如果编码时改成另一种顺序，最终 token 序列可能不同，模型看到的 ID 分布也会偏离训练语料。特殊 token 仍然要优先处理：用户指定的特殊 token 在编码时必须作为整体进入 ID 序列，不参与普通 byte 拆分。

解码方向更直接：把每个 token ID 查回 bytes，拼接成一个 bytestring，再调用 UTF-8 decode。这里必须考虑非法 byte 序列，因为用户可能传入任意 ID 列表，拼出来的 bytes 不一定是合法 UTF-8。工程上我会用 `errors="replace"`，让 malformed bytes 变成官方 replacement character `U+FFFD`，这样 decoder 不会因为坏输入崩掉。

| 边界条件 | 如果忽略会怎样 | 实现时我会怎么守住 |
|-|-|-|
| 特殊 token | 文档边界被 merge，结束符被拆成普通 token | 训练和编码都先按特殊 token 切分，并把特殊 token 作为固定 ID |
| pre-token 边界 | 跨词、跨标点产生奇怪 token，泛化变差 | 只在 regex pre-token 内部做 merge |
| merge 顺序 | 同一文本在训练和推理时得到不同 token ID | 编码时严格按 `merges` 的创建顺序应用 |
| 大文件编码 | 一次性读入导致内存线性增长 | 实现 `encode_iterable`，按安全边界流式产生 ID |
| 非法 UTF-8 | decode 抛异常，调试和评估被中断 | bytes decode 使用 `errors="replace"` |

### Demo 实现：

主要用于理解其原理

```python
from collections.abc import Iterator
import regex as re
import os
import pickle

from llm.args import get_parser


class BpeTokenizer:
    def __init__(self, special_tokens: list[str] | None = None, errors="replace"):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
        self.errors = errors
        self.vcab2id = dict[bytes, int]()
        self.id2vcab = dict[int, bytes]()
        self.merges = list[tuple[bytes, bytes]]()
        self.merge_ranks = dict[tuple[bytes, bytes], int]()
        self.special_tokens = sorted(
            [s.encode("utf-8", errors) for s in special_tokens] if special_tokens else [], key=len, reverse=True
        )

    def from_pretrained(
        self,
        id2vcab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | list[bytes] | None = None,
    ):
        self.id2vcab = id2vcab
        self.merges = merges
        if special_tokens:

            def is_list_of_str(lst):
                return isinstance(lst, list) and all(isinstance(item, str) for item in lst)

            if is_list_of_str(special_tokens):
                self.special_tokens = [s.encode("utf-8", self.errors) for s in special_tokens]
            else:
                self.special_tokens = special_tokens
        self.vcab2id = {v: k for k, v in self.id2vcab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def _pre_token(self, corpus: bytes) -> list[bytes]:
        if not self.special_tokens:
            return [
                match.group(0).encode("utf-8", self.errors)
                for match in re.finditer(self.pattern, corpus.decode("utf-8", self.errors))
            ]

        pattern = b"|".join(map(re.escape, self.special_tokens))
        parts = re.split(b"(" + pattern + b")", corpus)

        final_parts = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                final_parts.append(part)
            else:
                final_parts.extend(
                    [
                        match.group(0).encode("utf-8", self.errors)
                        for match in re.finditer(self.pattern, part.decode("utf-8", self.errors))
                    ]
                )
        return final_parts

    def encode(self, text: str) -> list[int]:
        bs = text.encode("utf-8", self.errors)
        pre_tokens = self._pre_token(bs)

        token_ids = []
        for pre_token in pre_tokens:
            if pre_token in self.special_tokens:
                token_ids.append(self.vcab2id[pre_token])
                continue

            tokens = tuple(bytes([c]) for c in pre_token)
            while len(tokens) > 1:
                pairs = list(zip(tokens[:-1], tokens[1:]))
                # Find the merge with the lowest rank
                rank = float("inf")
                best_pair_idx = -1
                for i, pair in enumerate(pairs):
                    if pair in self.merge_ranks and self.merge_ranks[pair] < rank:
                        rank = self.merge_ranks[pair]
                        best_pair_idx = i

                if best_pair_idx == -1:
                    break

                # Merge the best pair
                new_tokens = []
                if best_pair_idx > 0:
                    new_tokens.extend(tokens[:best_pair_idx])
                new_tokens.append(tokens[best_pair_idx] + tokens[best_pair_idx + 1])
                if best_pair_idx + 2 < len(tokens):
                    new_tokens.extend(tokens[best_pair_idx + 2 :])
                tokens = tuple(new_tokens)

            for vcab in tokens:
                token_ids.append(self.vcab2id[vcab])
        return token_ids

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        vcabs = [self.id2vcab.get(i, b"\xef\xbf\xbd") for i in token_ids]
        bs = b"".join(vcabs)
        return bs.decode("utf-8", self.errors)

    def save(self, out: str) -> None:
        obj = {"merge": self.merges, "id2vcab": self.id2vcab, "special_tokens": self.special_tokens}
        with open(out, "wb") as f:
            pickle.dump(obj, f)

    def load(self, ins: str):
        with open(ins, "rb") as f:
            obj = pickle.load(f)
            merges = obj["merge"]
            id2vcab = obj["id2vcab"]
            special_tokens = obj["special_tokens"]
        self.from_pretrained(id2vcab=id2vcab, merges=merges, special_tokens=special_tokens)

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], verbose=False):
        """
        Train the BPE tokenizer on the given corpus.

        :param input_path: Path to the training data.
        :param vocab_size: The desired vocabulary size.
        :param special_tokens: Special tokens to add to the vocabulary first.
        """
        self.special_tokens = sorted(
            [s.encode("utf-8", self.errors) for s in special_tokens] if special_tokens else [],
            key=len,
            reverse=True,
        )

        with open(input_path) as f:
            corpus = f.read().encode("utf-8", self.errors)

        for token in self.special_tokens:
            self.vcab2id[token] = len(self.vcab2id)

        for i in range(256):
            self.vcab2id[bytes([i])] = len(self.vcab2id)

        pre_tokens = self._pre_token(corpus)

        word_cnt: dict[tuple[bytes, ...], int] = {}
        for pre_token in pre_tokens:
            if pre_token in self.special_tokens:
                continue

            bs = tuple(bytes([b]) for b in pre_token)
            if not bs:
                continue

            word_cnt[bs] = word_cnt.get(bs, 0) + 1

        if verbose:
            print(f"cnt of distinct word: {len(word_cnt)}")

        pair_cnt: dict[tuple[bytes, bytes], int] = {}
        for word, cnt in word_cnt.items():
            for pair in zip(word[:-1], word[1:]):
                pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt

        def update_pair_counts(
            word: tuple[bytes, ...],
            pair_cnt: dict[tuple[bytes, bytes], int],
            cnt: int,
            sign=1,
        ):
            for pair in zip(word[:-1], word[1:]):
                pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt * sign
                if pair_cnt.get(pair, 0) <= 0:
                    del pair_cnt[pair]

        def update_word_counts(
            word_cnt: dict[tuple[bytes, ...], int],
            old_word: tuple[bytes, ...],
            new_word: tuple[bytes, ...],
            cnt: int,
        ):
            word_cnt[old_word] -= cnt
            if word_cnt[old_word] <= 0:
                del word_cnt[old_word]

            word_cnt[new_word] = word_cnt.get(new_word, 0) + cnt

        while len(self.vcab2id) < vocab_size:
            if verbose:
                print(f"vocab_size = {len(self.vcab2id)}, target {vocab_size}")

            if not pair_cnt:
                break

            best_pair = max(pair_cnt.keys(), key=lambda p: (pair_cnt.get(p, 0), p))

            self.vcab2id[best_pair[0] + best_pair[1]] = len(self.vcab2id)
            self.merges.append(best_pair)

            merged = False

            for word, cnt in list(word_cnt.items()):
                # word may have been removed or merged into another entry earlier in this round.
                if word_cnt.get(word, 0) < cnt:
                    continue

                new_word_list = []
                i = 0

                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word_list.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_word_list.append(word[i])
                        i += 1

                new_word = tuple(new_word_list)

                if word != new_word:
                    merged = True
                    update_pair_counts(word, pair_cnt, cnt, -1)
                    update_pair_counts(new_word, pair_cnt, cnt, 1)
                    update_word_counts(word_cnt, word, new_word, cnt)

            if not merged:
                break

        self.id2vcab = {id: vcab for vcab, id in self.vcab2id.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        return self.id2vcab, self.merges


if __name__ == "__main__":
    # train_data_file = os.path.join("data", "TinyStoriesV2-GPT4-train.txt")
    import numpy as np

    parser = get_parser()
    args = parser.parse_args()
    vocab_size = args.vocab_size
    tokenizer = BpeTokenizer(special_tokens=["<|endoftext|>"])
    print("start traning")
    tokenizer.train(args.train_source_file, vocab_size=vocab_size, special_tokens=["<|endoftext|>"], verbose=True)
    print("end traning")
    tokenizer.save(args.tokenizer_checkpoint)
    print(f"vocab size: {len(tokenizer.vcab2id)}")

    with open(args.train_source_file) as f:
        print("starting encoding train text to token ids")
        token_ids = tokenizer.encode(f.read())
        print("start persisting train tokens ids")
        np.save(args.train_data, np.array(token_ids))

    with open(args.valid_source_file) as f:
        print("starting encoding valid text to token ids")
        token_ids = tokenizer.encode(f.read())
        print("start persisting valid tokens ids")
        np.save(args.val_data, np.array(token_ids))

```

### 工业级实现：

工业界常使用底层语言实现BPE，比如 transformers 库中 HF 使用了 Rust 来实现高性能的 BPE 分词，相比上述使用python 实现的 demo 版本，主要有两个个动机：

- **更简单的并发管理**：使用原生支持多线程的语言，能够在计算密集的分词场景充分发挥多核性能，同时简化编程模型。
- **更高效的存储管理**：使用底层无GC、可自主管理内存的语言，能够在内存消耗较高场景更高效。
- **更极致的性能优化：**底层语言能够更方便的实现 SIMD 指令加速，merges 表常驻 L3 cache等。
