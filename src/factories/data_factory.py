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
        embedding_path='datasets/en_fasttext_word2vec_V100D20.json',
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
        with open(CONFIG_PATH) as f:
            corpus_dict = yaml.load(f, Loader=yaml.FullLoader)[self.dataset]

        language_id = corpus_dict['language']
        corpus_config = CorpusConfig(
            self.dataset,
            path=corpus_dict['path'],
            language_config=LANGUAGE_CONFIGS[language_id],
            maxlen=corpus_dict.get('maxlen', self.maxlen),
            vocab_size=corpus_dict.get('vocab_size', self.maxlen),
        )
        if not ('train' in corpus_config.path and all(os.path.isfile(p) for p in corpus_config.path.values())):
            raise KeyError  # TODO else warning?

        return Preprocessor(corpus_config).preprocess()
