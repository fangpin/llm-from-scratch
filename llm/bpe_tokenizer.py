from collections.abc import Iterator
import regex as re
import os


class BpeTokenizer:
    def __init__(self, special_tokens: list[str] | None = None, errors="replace"):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s"""
        self.errors = errors
        self.vcab2id = dict[bytes, int]()
        self.id2vcab = dict[int, bytes]()
        self.merges = list[tuple[bytes, bytes]]()
        self.special_tokens = sorted(
            [s.encode("utf-8", errors) for s in special_tokens] if special_tokens else [], key=len, reverse=True
        )

    def from_pretrained(
        self,
        id2vcab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id2vcab = id2vcab
        self.merges = merges
        if special_tokens:
            self.special_tokens = [s.encode("utf-8", self.errors) for s in special_tokens]
        self.vcab2id = {v: k for k, v in self.id2vcab.items()}

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
            else:
                tokens = tuple(bytes([c]) for c in pre_token)
                while True:
                    pair_cnt = dict[tuple[bytes, bytes], int]()
                    for pair in zip(tokens[:-1], tokens[1:]):
                        pair_cnt[pair] = pair_cnt.get(pair, 0) + 1

                    if not pair_cnt:
                        break

                    # Find the merge with the lowest rank
                    best_pair = min(pair_cnt, key=lambda p: self.merges.index(p) if p in self.merges else float("inf"))

                    if best_pair not in self.merges:
                        break

                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                            new_tokens.append(best_pair[0] + best_pair[1])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
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

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
        """
        Train the BPE tokenizer on the given corpus.

        :param corpus: A binary file-like object containing the training data.
        :param vocab_size: The desired vocabulary size.
        """
        self.special_tokens = sorted(
            [s.encode("utf-8", self.errors) for s in special_tokens] if special_tokens else [], key=len, reverse=True
        )

        corpus = list[bytes]()
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

        pair_cnt = dict[tuple[bytes, bytes], int]()
        for word, cnt in word_cnt.items():
            for pair in zip(word[:-1], word[1:]):
                pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt

        def update_pair_counts(word: tuple[bytes, ...], pair_cnt: dict[tuple[bytes, bytes], int], cnt: int, sign=1):
            for pair in zip(word[:-1], word[1:]):
                pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt * sign
                if pair_cnt.get(pair, 0) <= 0:
                    del pair_cnt[pair]

        while len(self.vcab2id) < vocab_size:
            best_pair = max(pair_cnt.keys(), key=lambda p: (pair_cnt.get(p, 0), p))
            self.vcab2id[best_pair[0] + best_pair[1]] = len(self.vcab2id)
            self.merges.append(best_pair)

            new_word_cnt: dict[tuple[bytes, ...], int] = {}
            merged = False
            for word, cnt in word_cnt.items():
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

                new_word_cnt[new_word] = new_word_cnt.get(new_word, 0) + cnt

            if not merged:
                break
            word_cnt = new_word_cnt

        self.id2vcab = {id: vcab for vcab, id in self.vcab2id.items()}
        return self.id2vcab, self.merges
