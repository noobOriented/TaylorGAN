import pathlib

import pytest

from .. import CorpusConfig, Segmentor, Tokenizer
from .._segmentor import SplitEnglish


@pytest.fixture(scope='session')
def cache_root_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('cache')


@pytest.fixture(scope='session', autouse=True)
def redirect_cache_root(cache_root_dir):
    from core.cache import cache_center
    cache_center.root_path = cache_root_dir


@pytest.fixture(scope='session')
def corpus_config(data_dir: pathlib.Path):
    return CorpusConfig(
        name='test',
        path=data_dir / 'train.txt',
        maxlen=10,
        embedding_path=data_dir / 'en_fasttext_word2vec_V100D20.json',
        segmentor=Segmentor(split_token=' ', operators=[SplitEnglish(type='split-english')]),
    )


class TestTokenizer:

    @pytest.fixture(scope='class')
    def tokenizer(self, corpus_config):
        return Tokenizer.fit_corpus(corpus_config)

    def test_mapping_consistent(self, tokenizer, corpus_config):
        with open(corpus_config.path['train'], 'r') as f:
            line = f.readline()
            ids1 = tokenizer.text_to_ids(line)
            text1 = tokenizer.ids_to_text(ids1)
            ids2 = tokenizer.text_to_ids(text1)
            text2 = tokenizer.ids_to_text(ids2)

        assert text1 == text2
        assert ids1 == ids2
