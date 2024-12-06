from __future__ import annotations

import collections
import dataclasses
import itertools
import pathlib
import typing as t

import more_itertools
import numpy as np
import numpy.typing as npt

from core.cache import cache_center
from library.utils import JSONSerializableMixin, format_path, logging_indent, tqdm_open

from ._config_objects import CorpusConfig, LanguageConfig, SpecialTokenConfig


class Preprocessor:

    def __init__(self, maxlen: int | None = None, vocab_size: int | None = None):
        self.maxlen = maxlen
        self.vocab_size = vocab_size

    @t.overload
    def preprocess(self, corpus_config: CorpusConfig, return_meta: t.Literal[True]) -> tuple[dict[str, TextDataset], MetaData]:
        ...

    @t.overload
    def preprocess(self, corpus_config: CorpusConfig, return_meta: bool = False) -> dict[str, TextDataset]:
        ...

    def preprocess(self, corpus_config: CorpusConfig, return_meta: bool = False):
        with logging_indent("Prepare text tokenizer..."):
            @cache_center.to_json(self.get_cache_dir(corpus_config) / 'tokenizer.json')
            def create_tokenizer():
                print(f'Build text mapper based on corpus data from {format_path(corpus_config.path["train"])}')
                return Tokenizer.fit_corpus(
                    corpus_config,
                    maxlen=self.maxlen,
                    vocab_size=self.vocab_size,
                )

            tokenizer = create_tokenizer()

        with logging_indent("Preprocess text corpus..."):
            data_collection: dict[str, TextDataset] = {}
            for key, path in corpus_config.path.items():
                @cache_center.to_npz(self.get_cache_dir(corpus_config) / f'{key}_data.npz')
                def _process_text_file(filepath):
                    print(f"Load corpus data from {format_path(filepath)}")
                    with tqdm_open(filepath) as f:
                        return tokenizer.texts_to_array(f)

                with logging_indent(f"{key} data:", bullet=False):
                    ids = _process_text_file(path)
                    texts = list(map(tokenizer.ids_to_text, ids))
                    text_dataset = TextDataset(ids=ids, texts=texts)
                    data_collection[key] = text_dataset

        if return_meta:
            metadata = MetaData(
                tokenizer=tokenizer,
                corpus_config=corpus_config,
                cache_dir=self.get_cache_dir(corpus_config),
            )
            return data_collection, metadata
        return data_collection

    def get_cache_dir(self, corpus_config: CorpusConfig):
        items = ["uttut"]
        if self.maxlen:
            items.append(f"L{self.maxlen}")
        if self.vocab_size:
            items.append(f"V{self.vocab_size}")
        return pathlib.Path(corpus_config.name, "_".join(items))


@dataclasses.dataclass
class TextDataset:
    ids: npt.NDArray[np.int_]
    texts: t.Sequence[str]

    def __len__(self):
        return len(self.ids)


@dataclasses.dataclass
class MetaData:
    tokenizer: Tokenizer
    corpus_config: CorpusConfig
    cache_dir: pathlib.Path

    def load_pretrained_embeddings(self) -> npt.NDArray[np.floating]:

        @cache_center.to_npz(self.cache_dir / 'word_vecs.npz')
        def load_embeddings():
            word_vec_config = self.corpus_config.language_config.load_pretrained_embeddings_msg()
            return word_vec_config.get_matrix_of_tokens(self.tokenizer.tokens)

        with logging_indent("Load pretrained embeddings:"):
            embeddings = load_embeddings()
            print(f"Dimensions: {embeddings.shape[1]}.")

        return embeddings

    @property
    def eos_idx(self) -> int:
        return self.tokenizer.eos_idx

    @property
    def special_token_config(self):
        return self.tokenizer.special_token_config


class Tokenizer(JSONSerializableMixin):

    special_token_config = SpecialTokenConfig(
        sos='<sos>',
        eos='</s>',
        pad='<pad>',
        unk='<unk>',
    )
    eos_idx = special_token_config.eos.idx

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
        vocab_size = vocab_size or corpus_config.vocab_size
        if maxlen:
            token_freq = collections.Counter(more_itertools.flatten(
                more_itertools.take(maxlen, sen)
                for sen in corpus_config.iter_train_sentences()
            ))
        else:
            token_freq, maxlen = _get_freq_and_maxlen(corpus_config.iter_train_sentences())

        all_tokens = more_itertools.unique_everseen(itertools.chain(
            cls.special_token_config.tokens,
            [token for token, _ in token_freq.most_common()],
        ))
        return cls(
            tokens=more_itertools.take(n=vocab_size, iterable=all_tokens),
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


def _get_freq_and_maxlen[T](sentences: t.Iterable[t.Sequence[T]]) -> tuple[collections.Counter[T], int]:
    freq = collections.Counter()
    maxlen = 0
    for sen in sentences:
        freq += collections.Counter(sen)
        maxlen = max(len(sen), maxlen)
    return freq, maxlen
