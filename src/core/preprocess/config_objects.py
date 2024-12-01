import os
import typing as t

import more_itertools

from library.utils import JSONSerializableMixin, format_path, logging_indent, tqdm_open
from uttut.pipeline.ops import CharTokenizer, MergeWhiteSpaceCharacters, StripWhiteSpaceCharacters

from .adaptors import UttutPipeline, WordEmbeddingCollection


class LanguageConfig(JSONSerializableMixin):

    def __init__(
        self,
        embedding_path: str | None,
        segmentor: UttutPipeline[str, list[str]] | None = None,
        split_token: str = '',
    ):
        self.embedding_path = embedding_path
        if segmentor is None:
            segmentor = UttutPipeline([
                MergeWhiteSpaceCharacters(),
                StripWhiteSpaceCharacters(),
                CharTokenizer(),
            ])

        self._segmentor = segmentor
        self.split_token = split_token

    def segmentize_text(self, text: str) -> list[str]:
        return self._segmentor.transform_sequence(text)

    def load_pretrained_embeddings_msg(self):
        if not (self.embedding_path and os.path.isfile(self.embedding_path)):
            raise FileNotFoundError(f"invalid embedding_path: {self.embedding_path}")
        print(f"Load pretrained embedding from : {format_path(self.embedding_path)}")
        return WordEmbeddingCollection.load_msg(self.embedding_path)

    def get_config(self):
        return {
            'embedding_path': str(self.embedding_path),
            'segmentor': self._segmentor.serialize(),
            'split_token': self.split_token,
        }

    @classmethod
    def from_config(cls, config_dict):
        config_dict['segmentor'] = UttutPipeline.deserialize(config_dict['segmentor'])
        return cls(**config_dict)


class CorpusConfig:

    def __init__(
        self,
        path: str | t.Mapping[str, str],
        language_config: LanguageConfig,
        maxlen: int | None = None,  # used when preprocessor.maxlen = None
        vocab_size: int | None = None,  # used when preprocessor.vocab_size = None
    ):
        if isinstance(path, str):
            path = {'train': path}
        self.path = path
        self.language_config = language_config
        self.maxlen = maxlen
        self.vocab_size = vocab_size

    def iter_train_sentences(self) -> t.Iterator[list[str]]:
        with tqdm_open(self.path['train']) as it:
            for s in it:
                yield self.language_config.segmentize_text(s)

    def is_valid(self) -> bool:
        return 'train' in self.path and all(os.path.isfile(p) for p in self.path.values())


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
