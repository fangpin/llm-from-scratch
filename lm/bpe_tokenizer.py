import typing
import os
from typing import List, Tuple, BinaryIO
import regex as re


class BpeTokenizer:
    def __init__(self, errors="replace"):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.errors = errors
        self.vcab = {}
        self.merges = []

    def train(self, corpus: BinaryIO, vocab_size: int, special_tokens: List[bytes]):
        """
        Train the BPE tokenizer on the given corpus.

        :param corpus: A binary file-like object containing the training data.
        :param vocab_size: The desired vocabulary size.
        """
        for i in range(256):
            self.vcab[bytes([i])] = i
        n = 256
        word_cnt = {}
        for word in re.finditer(
            self.pattern, corpus.read().decode("utf-8", self.errors)
        ):
            bs = tuple((i,) for i in tuple(word.string.encode("utf-8", self.errors)))
            word_cnt[bs] = word_cnt.get(bs, 0) + 1

        while True:
            pair_cnt = {}
            for word, cnt in word_cnt.items():
                for a, b in zip(word[-1], word[1:]):
                    pair = a + b
                    pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt

            cnt, pair = max(((cnt, pair) for pair, cnt in pair_cnt))
            self.vcab[pair] = n
            self.merges.append(pair)
            n += 1
            if n + len(special_tokens) == vocab_size:
                break

        return self.vcab, self.merges
