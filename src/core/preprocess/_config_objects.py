import pathlib
import re
import typing as t
import unicodedata

import more_itertools
import numpy as np
import numpy.typing as npt
import pydantic

from library.utils import logging_indent, tqdm_open


class WordEmbeddingCollection(pydantic.BaseModel):
    token2index: dict[str, int]
    vectors: list[list[float]]

    UNK: t.ClassVar = '<unk>'

    def get_matrix_of_tokens(self, tokens: t.Iterable[str]) -> npt.NDArray[np.float32]:
        return np.asarray(
            [self._get_vector_of_token(token) for token in tokens],
            dtype=np.float32,
        )

    def _get_vector_of_token(self, token: str):
        if (index := self.token2index.get(token)) is not None:
            return self.vectors[index]
        if (index := self.token2index.get(self.UNK)) is not None:
            return self.vectors[index]
        return np.zeros_like(self.vectors[0])


class Segmentor(pydantic.BaseModel):
    split_token: str = ' '

    def segmentize_text(self, s: str) -> list[str]:
        # TODO serializable?
        s = re.sub(r'\s+', ' ', s)
        s = s.strip()
        s = s.lower()
        orig_tokens = s.split(self.split_token)
        split_tokens = more_itertools.flatten(map(_run_split_on_punc, orig_tokens))
        return self.split_token.join(split_tokens).split(self.split_token)

    def join_text(self, texts: list[str]) -> str:
        return self.split_token.join(texts)


class CorpusConfig(pydantic.BaseModel):
    name: str
    path: dict[str, pathlib.Path]
    segmentor: Segmentor
    embedding_path: pathlib.Path
    maxlen: int | None = None  # used when preprocessor.maxlen = None
    vocab_size: int | None = None  # used when preprocessor.vocab_size = None

    @pydantic.field_validator('path', mode='before')
    @classmethod
    def _parse_path_str(cls, v):
        return v if isinstance(v, t.Mapping) else {'train': v}

    def iter_train_sentences(self) -> t.Iterator[list[str]]:
        with tqdm_open(self.path['train']) as it:
            for s in it:
                yield self.segmentor.segmentize_text(s)

    @property
    def cache_path(self):
        items = ["uttut"]
        if self.maxlen:
            items.append(f"L{self.maxlen}")
        if self.vocab_size:
            items.append(f"V{self.vocab_size}")
        return pathlib.Path(self.name, "_".join(items))



class SpecialTokenConfig:

    class TokenIdxTuple(t.NamedTuple):
        token: str
        idx: int

    def __init__(self, **kwargs: str):
        if not more_itertools.all_unique(kwargs.values()):
            raise KeyError("special tokens conflict.")

        self._token_list = list(kwargs.values())
        self._attrs = {
            key: self.TokenIdxTuple(token, idx)
            for idx, (key, token) in enumerate(kwargs.items())
        }

    @property
    def tokens(self) -> list[str]:
        return self._token_list

    def __getattr__(self, key):
        if key in self._attrs:
            return self._attrs[key]
        raise AttributeError

    def summary(self):
        with logging_indent("Special tokens config:"):
            for key, (token, idx) in self._attrs.items():
                print(f"{key} token: '{token}', index: {idx}.")


def _run_split_on_punc(text: str) -> list[str]:
    """
    Recognize punctuations and seperate them into independent tokens.

    E.g.
    1. "abc, cdf" -> ["abc", ",", " ", "cdf"]
    2. "I like apples." -> ["I", "like", "apples", "."]

    """
    start_new_word = True
    output: list[list[str]] = []
    for c in text:
        if _is_punctuation(c):
            output.append([])
            start_new_word = True
        elif start_new_word:
            output.append([])
            start_new_word = False

        output[-1].append(c)

    return [''.join(x) for x in output]


def _is_punctuation(char) -> bool:
    """Checks whether `chars` is a punctuation character.

    We treat all non-letter/number ASCII as punctuation.
    Characters such as "^", "$", and "`" are not in the Unicode
    Punctuation class but we treat them as punctuation anyways, for consistency.

    """
    return ord(char) in _ASCII_PUNCTUATIONS or unicodedata.category(char).startswith('P')


_ASCII_PUNCTUATIONS = {*range(33, 48), *range(58, 65), *range(91, 97), *range(123, 127)}
