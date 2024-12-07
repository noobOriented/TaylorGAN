import typing as t

import pydantic
import yaml

from core.preprocess import CorpusConfig, MetaData, Preprocessor, TextDataset
from library.utils import format_id


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
        d = {'name': self.dataset, 'maxlen': self.maxlen, 'vocab_size': self.vocab_size}
        with open(CONFIG_PATH) as f:
            d |= yaml.safe_load(f)[self.dataset]
    
        corpus_config = CorpusConfig(**d)
        if not ('train' in corpus_config.path and all(p.exists() for p in corpus_config.path.values())):
            raise KeyError  # TODO else warning?

        return Preprocessor(corpus_config).preprocess()
