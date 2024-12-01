import abc
import pathlib
import typing as t
from more_itertools import with_iter

from core.cache import cache_center
from library.utils import format_path, logging_indent, tqdm_open

from .config_objects import CorpusConfig
from .record_objects import MetaData, TextDataset
from .tokenizers import Tokenizer


class Preprocessor(abc.ABC):

    @abc.abstractmethod
    def preprocess(self, corpus_config: CorpusConfig) -> MetaData:
        pass


class UttutPreprocessor(Preprocessor):

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
                print(f'Build text mapper based on corpus data from {format_path(corpus_config.path.train)}')
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
                    return tokenizer.texts_to_array(with_iter(tqdm_open(filepath)))

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
