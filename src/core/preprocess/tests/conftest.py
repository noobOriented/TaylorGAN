import pytest

from ..config_objects import CorpusConfig, LanguageConfig
from library.utils import NamedObject


@pytest.fixture(scope='session')
def language_config(data_dir):
    return LanguageConfig(
        embedding_path=data_dir / 'en_fasttext_word2vec_V100D20.msg',
        split_token=' ',
    )


@pytest.fixture(scope='session')
def corpus_config(data_dir, language_config):
    return NamedObject(
        CorpusConfig(
            path=data_dir / 'train.txt',
            maxlen=10,
            language_config=language_config,
        ),
        name='test',
    )
