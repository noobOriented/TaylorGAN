import os
import typing as t

import pydantic
import yaml
from dotenv import load_dotenv

from core.preprocess import CorpusConfig, LanguageConfig, MetaData, TextDataset, Preprocessor
from library.utils import format_id


load_dotenv('.env')

CONFIG_PATH = 'datasets/corpus.yaml'
LANGUAGE_CONFIGS = {
    'english': LanguageConfig(
        embedding_path=os.getenv('PRETRAINED_EN_WORD_FASTTEXT_PATH'),
        split_token=' ',
    ),
    'test': LanguageConfig(
        embedding_path='datasets/en_fasttext_word2vec_V100D20.msg',
        split_token=' ',
    ),
}


class DataConfigs(pydantic.BaseModel):
    dataset: t.Annotated[str, pydantic.Field(description='the choice of corpus.')]
    maxlen: t.Annotated[int | None, pydantic.Field(ge=1, description='the max length of sequence padding.')] = None
    vocab_size: t.Annotated[
        int | None,
        pydantic.Field(ge=1, description='the maximum number of tokens. ordered by descending frequency.'),
    ] = None

    def load_data(self) -> tuple[dict[str, TextDataset], MetaData]:
        print(f"data_id: {format_id(self.dataset)}")
        preprocessor = Preprocessor(self.maxlen, self.vocab_size)
        corpus_config = _load_corpus_table(CONFIG_PATH)[self.dataset]
        return preprocessor.preprocess(corpus_config, return_meta=True)


def _load_corpus_table(path):
    corpus_table: dict[str, CorpusConfig] = {}
    with open(path) as f:
        for data_id, corpus_dict in yaml.load(f, Loader=yaml.FullLoader).items():
            language_id = corpus_dict['language']
            config = CorpusConfig(
                data_id,
                path=corpus_dict['path'],
                language_config=LANGUAGE_CONFIGS[language_id],
                maxlen=corpus_dict.get('maxlen'),
                vocab_size=corpus_dict.get('vocab_size'),
            )
            if 'train' in config.path and all(os.path.isfile(p) for p in config.path.values()):
                # TODO else warning?
                corpus_table[data_id] = config

    return corpus_table
