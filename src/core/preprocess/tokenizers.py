import abc
import collections
import itertools
import typing as t

import more_itertools
import numpy as np
import numpy.typing as npt

from library.utils import JSONSerializableMixin, logging_indent

from .config_objects import CorpusConfig, LanguageConfig, SpecialTokenConfig


class Tokenizer(abc.ABC, JSONSerializableMixin):

    special_token_config = SpecialTokenConfig(
        sos='<sos>',
        eos='</s>',
        pad='<pad>',
        unk='<unk>',
    )
    eos_idx = np.int32(special_token_config.eos.idx)

    def __init__(self, tokens: list[str], language_config: LanguageConfig, maxlen: int):
        self.tokens = tokens
        self.maxlen = maxlen
        self.language_config = language_config
        self._token2index = {token: i for i, token in enumerate(tokens)}

    def texts_to_array(self, texts: t.Iterable[str]) -> npt.NDArray[np.int32]:
        return np.asarray(
            [self.text_to_ids(s) for s in texts],
            dtype=np.int32,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def summary(self):
        with logging_indent(f"{self.__class__.__name__} summary:"):
            print(f"Maxlen: {self.maxlen}.")
            print(f"Vocabulary size: {self.vocab_size}.")
            self.special_token_config.summary()

    def text_to_ids(self, text: str) -> list[int]:
        tokens = self.language_config.segmentize_text(text)
        tokens.append(self.special_token_config.eos.token)
        tokens = more_itertools.padded(tokens, self.special_token_config.pad.token)
        tokens = more_itertools.take(self.maxlen, tokens)
        return [
            self._token2index.get(s, self.special_token_config.unk.idx)
            for s in tokens
        ]

    def ids_to_text(self, ids: t.Sequence[int]) -> str:
        tokens = [self.tokens[idx] for idx in itertools.takewhile(lambda x: x != self.eos_idx, ids)]
        return self.language_config.split_token.join(tokens)

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
