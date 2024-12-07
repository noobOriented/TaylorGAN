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
from library.utils import format_path, logging_indent, tqdm_open

from ._config_objects import (
    CorpusConfig, Segmentor, SpecialTokenConfig, WordEmbeddingCollection,
)


class Preprocessor:

    def __init__(self, corpus_config: CorpusConfig):
        self.corpus_config = corpus_config

    def preprocess(self):
        with logging_indent("Prepare text tokenizer..."):
            tokenizer = self._create_tokenizer()

        with logging_indent("Preprocess text corpus..."):
            data_collection: dict[str, TextDataset] = {}
            for key, path in self.corpus_config.path.items():
                @cache_center.to_npz(self.corpus_config.cache_path / f'{key}_data.npz')
                def _process_text_file(filepath):
                    print(f"Load corpus data from {format_path(filepath)}")
                    with tqdm_open(filepath) as f:
                        return tokenizer.texts_to_array(f)

                with logging_indent(f"{key} data:", bullet=False):
                    ids = _process_text_file(path)
                    texts = [tokenizer.ids_to_text(idx) for idx in ids]
                    text_dataset = TextDataset(ids=ids, texts=texts)
                    data_collection[key] = text_dataset

        metadata = MetaData(
            tokenizer=tokenizer,
            embedding_path=self.corpus_config.embedding_path,
            cache_dir=self.corpus_config.cache_path,
        )
        return data_collection, metadata

    def _create_tokenizer(self):
        if cache_center.root_path:
            p = cache_center.root_path / self.corpus_config.cache_path / 'tokenizer.json'
            if p.exists():
                with open(p) as f:
                    return Tokenizer.model_validate_json(f.read())

        print(f'Build text mapper based on corpus data from {format_path(self.corpus_config.path["train"])}')
        tokenizer = Tokenizer.fit_corpus(self.corpus_config)
        if cache_center.root_path:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                f.write(tokenizer.model_dump_json(indent=2))
        return tokenizer


@dataclasses.dataclass
class TextDataset:
    ids: npt.NDArray[np.int_]
    texts: t.Sequence[str]

    def __len__(self):
        return len(self.ids)


@dataclasses.dataclass
class MetaData:
    tokenizer: Tokenizer
    embedding_path: pathlib.Path
    cache_dir: pathlib.Path

    def load_pretrained_embeddings(self) -> npt.NDArray[np.floating]:

        @cache_center.to_npz(self.cache_dir / 'word_vecs.npz')
        def load_embeddings():
            print(f"Load pretrained embedding from : {format_path(self.embedding_path)}")
            with open(self.embedding_path) as f:
                wv = WordEmbeddingCollection.model_validate_json(f.read())
            return wv.get_matrix_of_tokens(self.tokenizer.tokens)

        with logging_indent("Load pretrained embeddings:"):
            embeddings = load_embeddings()
            print(f"Dimensions: {embeddings.shape[1]}.")

        return embeddings


class Tokenizer(pydantic.BaseModel):
    tokens: list[str]
    segmentor: Segmentor
    maxlen: int

    special_token_config: t.ClassVar = SpecialTokenConfig(
        sos='<sos>',
        eos='</s>',
        pad='<pad>',
        unk='<unk>',
    )
    eos_idx: t.ClassVar[int] = special_token_config.eos.idx

    def texts_to_array(self, texts: t.Iterable[str]) -> npt.NDArray[np.int32]:
        return np.asarray(
            [self.text_to_ids(s) for s in texts],
            dtype=np.int32,
        )

    def text_to_ids(self, text: str) -> list[int]:
        tokens = self.segmentor.segmentize_text(text)
        tokens.append(self.special_token_config.eos.token)
        tokens = more_itertools.padded(tokens, self.special_token_config.pad.token)
        tokens = more_itertools.take(self.maxlen, tokens)
        return [
            self._token2index.get(s, self.special_token_config.unk.idx)
            for s in tokens
        ]

    def ids_to_text(self, ids: t.Sequence[int]) -> str:
        tokens = [self.tokens[idx] for idx in itertools.takewhile(lambda x: x != self.eos_idx, ids)]
        return self.segmentor.join_text(tokens)

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
            cls.special_token_config.tokens,
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
            self.special_token_config.summary()

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
