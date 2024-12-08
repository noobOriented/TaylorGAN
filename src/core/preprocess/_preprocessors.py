from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import pathlib
import typing as t

import more_itertools
import numpy as np
import numpy.typing as npt
import pydantic

from core.cache import cache_center
from library.utils import format_path, logging_indent

from ._configs import CorpusConfig, Segmentor, SpecialToken


@dataclasses.dataclass
class TextDataset:
    ids: npt.NDArray[np.int_]
    texts: t.Sequence[str]

    def __len__(self):
        return len(self.ids)


@dataclasses.dataclass
class PreprocessResult:
    dataset: dict[str, TextDataset]
    tokenizer: Tokenizer
    embedding_path: pathlib.Path
    cache_key: str

    @functools.cached_property
    def embedding_matrix(self) -> npt.NDArray[np.floating]:

        @cache_center.to_npz(self.cache_key, 'word_vecs.npz')
        def load_embeddings():
            print(f"Load pretrained embedding from : {format_path(self.embedding_path)}")
            with open(self.embedding_path) as f:
                d = pydantic.TypeAdapter(dict[str, list[float]]).validate_json(f.read())

            dimension = len(more_itertools.first(d.values()))
            unk_vector = d.get(self.special_tokens.UNK) or ([0.] * dimension)
            return np.asarray(
                [d.get(token, unk_vector) for token in self.tokenizer.tokens],
                dtype=np.float32,
            )

        with logging_indent("Load pretrained embeddings:"):
            embeddings = load_embeddings()
            print(f"Dimensions: {embeddings.shape[1]}.")

        return embeddings

    @property
    def special_tokens(self) -> type[SpecialToken]:
        return self.tokenizer.special_tokens

    def summary(self):
        with logging_indent("Data summary:"):
            for key, array in self.dataset.items():
                print(f"{key} data contains {len(array)} sentences.")

        self.tokenizer.summary()


class Tokenizer(pydantic.BaseModel):
    tokens: list[str]
    segmentor: Segmentor
    maxlen: int

    special_tokens: t.ClassVar = SpecialToken

    def texts_to_array(self, texts: t.Iterable[str]) -> npt.NDArray[np.int32]:
        return np.asarray([self.text_to_ids(s) for s in texts], dtype=np.int32)

    def text_to_ids(self, text: str):
        tokens = self.segmentor.segmentize_text(text)
        tokens.append(self.special_tokens.EOS.value)
        tokens = more_itertools.padded(tokens, self.special_tokens.PAD.value)
        tokens = more_itertools.take(self.maxlen, tokens)
        return [self._token2index.get(s, self.special_tokens.UNK.idx) for s in tokens]

    def ids_to_text(self, ids: t.Sequence[int], /) -> str:
        return self.segmentor.join_text(
            self.tokens[idx]
            for idx in itertools.takewhile(lambda x: x != self.special_tokens.EOS.idx, ids)
        )

    @classmethod
    def fit_corpus(cls, corpus_config: CorpusConfig):
        maxlen = corpus_config.maxlen
        vocab_size = corpus_config.vocab_size
        if maxlen:
            token_freq = collections.Counter(more_itertools.flatten(
                more_itertools.take(maxlen, sen)
                for sen in corpus_config.iter_train_sentences()
            ))
        else:
            token_freq, maxlen = _get_freq_and_maxlen(corpus_config.iter_train_sentences())

        all_tokens = more_itertools.unique_everseen(itertools.chain(
            cls.special_tokens,
            [token for token, _ in token_freq.most_common()],
        ))
        tokens = more_itertools.take(vocab_size, all_tokens) if vocab_size else list(all_tokens)
        return cls(
            tokens=tokens,
            segmentor=corpus_config.segmentor,
            maxlen=maxlen,
        )

    def summary(self):
        with logging_indent(f"{self.__class__.__name__} summary:"):
            print(f"Maxlen: {self.maxlen}.")
            print(f"Vocabulary size: {len(self.tokens)}.")
            with logging_indent("Special tokens config:"):
                for idx, token in self.special_tokens.__members__.items():
                    print(f"{token.name} token: '{token}', index: {idx}.")

    @functools.cached_property
    def _token2index(self):
        return {token: i for i, token in enumerate(self.tokens)}


def _get_freq_and_maxlen[T](sentences: t.Iterable[t.Sequence[T]], /) -> tuple[collections.Counter[T], int]:
    freq = collections.Counter()
    maxlen = 0
    for sen in sentences:
        freq += collections.Counter(sen)
        maxlen = max(len(sen), maxlen)
    return freq, maxlen
