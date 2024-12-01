import abc
import collections
import itertools
import typing as t

import more_itertools
import numpy as np
import numpy.typing as npt

from library.utils import JSONSerializableMixin, logging_indent
from uttut.pipeline.ops import Pad, Token2Index
from uttut.pipeline.ops.add_end_token import AddEndToken

from .adaptors import UttutPipeline
from .config_objects import CorpusConfig, LanguageConfig, SpecialTokenConfig


class Tokenizer(abc.ABC, JSONSerializableMixin):

    special_token_config = SpecialTokenConfig(
        sos='<sos>',
        eos='</s>',
        pad='<pad>',
        unk='<unk>',
    )
    eos_idx = np.int32(special_token_config.eos.idx)

    def __init__(self, tokens: t.Sequence[str], maxlen: int):
        self.tokens = tokens
        self.maxlen = maxlen

    def texts_to_array(self, texts: t.Iterable[str]) -> npt.NDArray[np.int32]:
        return np.asarray(
            [self.text_to_ids(s) for s in texts],
            dtype=np.int32,
        )

    @abc.abstractmethod
    def text_to_ids(self, text: str) -> list[int]:
        ...

    @abc.abstractmethod
    def ids_to_text(self, ids: t.Sequence[int], split: str | None = None) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def summary(self):
        with logging_indent(f"{self.__class__.__name__} summary:"):
            print(f"Maxlen: {self.maxlen}.")
            print(f"Vocabulary size: {self.vocab_size}.")
            self.special_token_config.summary()


class UttutTokenizer(Tokenizer):

    def __init__(self, tokens: list[str], language_config: LanguageConfig, maxlen: int):
        super().__init__(tokens, maxlen)
        self.language_config = language_config
        self._token_indexer = UttutPipeline[list[str], list[int]]([
            AddEndToken(self.special_token_config.eos.token),
            Pad(maxlen, pad_token=self.special_token_config.pad.token),
            Token2Index(
                {token: i for i, token in enumerate(tokens)},
                unk_token=self.special_token_config.unk.token,
            ),
        ])

    def text_to_ids(self, text: str) -> list[int]:
        tokens = self.language_config.segmentize_text(text)
        return self._token_indexer.transform_sequence(tokens)

    def ids_to_text(self, ids: t.Sequence[int], split: str | None = None) -> str:
        if split is None:
            split = self.language_config.split_token
        tokens = [self.tokens[idx] for idx in itertools.takewhile(lambda x: x != self.eos_idx, ids)]
        return split.join(tokens)

    @classmethod
    def fit_corpus(cls, corpus_config: CorpusConfig, maxlen: int | None = None, vocab_size: int | None = None):
        maxlen = maxlen or corpus_config.maxlen
        if maxlen:
            token_freq = collections.Counter(more_itertools.flatten(
                more_itertools.take(maxlen, sen)
                for sen in corpus_config.iter_train_sentences()
            ))
        else:
            token_freq, maxlen = get_freq_and_maxlen(corpus_config.iter_train_sentences())

        all_tokens = more_itertools.unique_everseen(itertools.chain(
            cls.special_token_config.tokens,
            [token for token, _ in token_freq.most_common()],
        ))
        return cls(
            tokens=more_itertools.take(n=vocab_size or corpus_config.vocab_size, iterable=all_tokens),
            language_config=corpus_config.language_config,
            maxlen=maxlen,
        )

    def get_config(self):
        return {
            'tokens': self.tokens,
            'language_config': self.language_config.get_config(),
            'maxlen': self.maxlen,
        }

    @classmethod
    def from_config(cls, config_dict):
        return cls(
            tokens=config_dict['tokens'],
            language_config=LanguageConfig.from_config(config_dict['language_config']),
            maxlen=config_dict['maxlen'],
        )


def get_freq_and_maxlen[T](sentences: t.Iterable[t.Sequence[T]]) -> tuple[collections.Counter[T], int]:
    freq = collections.Counter()
    maxlen = 0
    for sen in sentences:
        freq += collections.Counter(sen)
        maxlen = max(len(sen), maxlen)
    return freq, maxlen
