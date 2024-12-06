import pytest

from .. import CorpusConfig, LanguageConfig


@pytest.fixture(scope='session')
def cache_root_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('cache')


@pytest.fixture(scope='session', autouse=True)
def redirect_cache_root(cache_root_dir):
    from core.cache import cache_center
    cache_center.root_path = str(cache_root_dir)


@pytest.fixture(scope='session')
def language_config(data_dir):
    return LanguageConfig(
        embedding_path=data_dir / 'en_fasttext_word2vec_V100D20.msg',
        split_token=' ',
    )


@pytest.fixture(scope='session')
def corpus_config(data_dir, language_config):
    return CorpusConfig(
        'test',
        path=data_dir / 'train.txt',
        maxlen=10,
        language_config=language_config,
    )
