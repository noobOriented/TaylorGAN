import pathlib
import typing as t

import more_itertools
import numpy as np
import numpy.typing as npt
import pydantic

from library.utils import logging_indent, tqdm_open
from ._ops import Operator


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
    operators: list[Operator]

    def segmentize_text(self, s: str) -> list[str]:
        for op in self.operators:
            s = op(s)
        return s

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
