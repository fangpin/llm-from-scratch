from typing import List, Tuple, Dict, Iterator
import regex as re


class BpeTokenizer:
    def __init__(self, special_tokens: List[bytes], errors="replace"):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.errors = errors
        self.vcab2id = dict[bytes, int]()
        self.id2vcab = dict[int, bytes]()
        self.merges = list[Tuple[bytes, bytes]]()
        self.special_tokens = special_tokens

    def _pre_token(self, corpus: bytes) -> Iterator[bytes]:
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            pattern = f"({b'|'.join(escaped_tokens)})".encode("utf-8")
            return re.splititer(pattern, corpus)
        else:
            return iter([corpus])

    def encode(self, text: str) -> list[int]:
        bs = text.encode("utf-8", self.errors)
        pre_tokens = self._pre_token(bs)
        tokens = [tuple(bytes([c]) for c in b) for b in pre_tokens]
        for merge in self.merges:
            new_tokens = []
            for token in tokens:
                i = 0
                new_token = []
                while i < len(token):
                    if (
                        i < len(token) - 1
                        and token[i] == merge[0]
                        and token[i + 1] == merge[1]
                    ):
                        new_token.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_token.append(token[i])
                        i += 1
                new_tokens.append(new_token)
            tokens = new_tokens
        token_ids = list[int]()
        for token in tokens:
            for vcab in token:
                token_ids.append(self.vcab2id[vcab])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        vcabs = [self.id2vcab.get(i, b"\xef\xbf\xbd") for i in token_ids]
        bs = b"".join(vcabs)
        return bs.decode("utf-8", self.errors)

    def train(self, corpus: bytes, vocab_size: int):
        """
        Train the BPE tokenizer on the given corpus.

        :param corpus: A binary file-like object containing the training data.
        :param vocab_size: The desired vocabulary size.
        """
        for token in self.special_tokens:
            self.vcab2id[token] = len(self.vcab2id)

        for i in range(256):
            self.vcab2id[bytes([i])] = len(self.vcab2id)

        pre_tokens = self._pre_token(corpus)

        word_cnt: Dict[Tuple[bytes, ...], int] = {}
        for pre_token in pre_tokens:
            for word in re.finditer(
                self.pattern, pre_token.decode("utf-8", self.errors)
            ):
                bs = tuple(
                    bytes([b]) for b in word.group(0).encode("utf-8", self.errors)
                )
                if not bs:
                    continue
                word_cnt[bs] = word_cnt.get(bs, 0) + 1

        while len(self.vcab2id) < vocab_size:
            pair_cnt = dict[tuple[bytes, bytes], int]()
            for word, cnt in word_cnt.items():
                if len(word) < 2:
                    continue
                for pair in zip(word[:-1], word[1:]):
                    pair_cnt[pair] = pair_cnt.get(pair, 0) + cnt

            if not pair_cnt:
                break  # No more pairs to merge

            cnt, pair = max(((cnt, pair) for pair, cnt in pair_cnt.items()))
            self.vcab2id[pair[0] + pair[1]] = len(self.vcab2id)
            self.merges.append(pair)

            new_word_cnt: Dict[Tuple[bytes, ...], int] = {}
            for word, cnt in word_cnt.items():
                new_word = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == pair[0]
                        and word[i + 1] == pair[1]
                    ):
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_tuple = tuple(new_word)
                new_word_cnt[new_word_tuple] = new_word_cnt.get(new_word_tuple, 0) + cnt

            word_cnt = new_word_cnt
        self.id2vcab = {id: vcab for vcab, id in self.vcab2id.items()}
        return self.vcab2id, self.merges


if __name__ == "__main__":
    text = b"low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    special_tokens = [b"<|endoftext|>"]
    t = BpeTokenizer(special_tokens)
    t.train(text, 256 + 4)
    print(t.vcab2id)
    print("*" * 30)

    print(t.merges)
    print("*" * 30)

    string1 = "lower newest widest"
    token_ids = t.encode(string1)
    print(token_ids)
    print("*" * 30)

    string2 = t.decode(token_ids)
    print(string2)
    print("*" * 30)

    assert string1 == string2

    # takes about 30 mins
    # with open("TinyStoriesV2-GPT4-train.txt", "rb") as f:
    #     text = f.read()
    #     t = BpeTokenizer(special_tokens)
    #     t.train(text, 10000)
    #     print(t.vcab2id)
    #     print("*" * 30)
    #     print(t.merges)
