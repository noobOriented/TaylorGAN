import functools
import typing as t

import pydantic
import yaml

from core.cache import cache_center
from core.preprocess import CorpusConfig, MetaData, TextDataset, Tokenizer
from library.utils import format_id, format_path, logging_indent, tqdm_open


CONFIG_PATH = 'datasets/corpus.yaml'


class DataConfigs(pydantic.BaseModel):
    dataset: t.Annotated[str, pydantic.Field(description='the choice of corpus.')]
    maxlen: t.Annotated[int | None, pydantic.Field(ge=1, description='the max length of sequence padding.')] = None
    vocab_size: t.Annotated[
        int | None,
        pydantic.Field(ge=1, description='the maximum number of tokens. ordered by descending frequency.'),
    ] = None

    def load_data(self) -> tuple[dict[str, TextDataset], MetaData]:
        print(f"data_id: {format_id(self.dataset)}")

        with logging_indent("Prepare text tokenizer..."):
            tokenizer = self._create_tokenizer()

        with logging_indent("Preprocess text corpus..."):
            data_collection: dict[str, TextDataset] = {}
            for key, path in self._corpus_config.path.items():
                @cache_center.to_npz(self._corpus_config.cache_path / f'{key}_data.npz')
                def _process_text_file(filepath):
                    print(f"Load corpus data from {format_path(filepath)}")
                    with tqdm_open(filepath) as f:
                        return tokenizer.texts_to_array(f)

                with logging_indent(f"{key} data:", bullet=False):
                    ids = _process_text_file(path)
                    texts = [tokenizer.ids_to_text(idx) for idx in ids]
                    data_collection[key] = TextDataset(ids=ids, texts=texts)

        metadata = MetaData(
            tokenizer=tokenizer,
            embedding_path=self._corpus_config.embedding_path,
            cache_dir=self._corpus_config.cache_path,
        )
        return data_collection, metadata

    def _create_tokenizer(self):
        if cache_center.root_path:
            p = cache_center.root_path / self._corpus_config.cache_path / 'tokenizer.json'
            if p.exists():
                with open(p) as f:
                    return Tokenizer.model_validate_json(f.read())

        print(f'Build text mapper based on corpus data from {format_path(self._corpus_config.path["train"])}')
        tokenizer = Tokenizer.fit_corpus(self._corpus_config)
        if cache_center.root_path:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                f.write(tokenizer.model_dump_json(indent=2))
        return tokenizer

    @functools.cached_property
    def _corpus_config(self):
        d = {'name': self.dataset, 'maxlen': self.maxlen, 'vocab_size': self.vocab_size}
        with open(CONFIG_PATH) as f:
            d |= yaml.safe_load(f)[self.dataset]
    
        c = CorpusConfig(**d)
        if not ('train' in c.path and all(p.exists() for p in c.path.values())):
            raise KeyError  # TODO else warning?
        return c
