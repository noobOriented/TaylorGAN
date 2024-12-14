import enum
import functools
import pathlib
import typing as t

import pydantic
import rich.progress

from ._segmentor import Segmentor


class CorpusConfig(pydantic.BaseModel):
    name: str
    path: dict[str, pathlib.Path]
    segmentor: Segmentor
    embedding_path: pathlib.Path
    maxlen: int | None = None
    vocab_size: int | None = None

    @pydantic.field_validator('path', mode='before')
    @classmethod
    def _parse_path_str(cls, v):
        return v if isinstance(v, t.Mapping) else {'train': v}

    def iter_train_sentences(self) -> t.Iterator[list[str]]:
        with rich.progress.open(self.path['train'], 'r') as it:
            for s in it:
                yield self.segmentor.segmentize_text(s)


class SpecialToken(enum.StrEnum):
    SOS = '<sos>'
    EOS = '</s>'
    PAD = '<pad>'
    UNK = '<unk>'

    @functools.cached_property
    def idx(self) -> int:
        return next(
            i
            for i, token in enumerate(self.__class__)
            if self == token
        )
