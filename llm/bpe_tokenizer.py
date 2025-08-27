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
            if verbose:
                print(f"vocab_size = {len(self.vcab2id)}, target {vocab_size}")
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
