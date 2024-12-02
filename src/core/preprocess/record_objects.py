import dataclasses
import pathlib
import typing as t

import numpy as np
import numpy.typing as npt

from core.cache import cache_center
from core.preprocess.tokenizers import Tokenizer
from library.utils import logging_indent

from .config_objects import CorpusConfig


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
        return int(self.tokenizer.eos_idx)

    @property
    def special_token_config(self):
        return self.tokenizer.special_token_config
