import numpy as np
import pytest
import pathlib

from .. import CorpusConfig, LanguageConfig, Preprocessor, Tokenizer
from .._config_objects import WordEmbeddingCollection


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
        embedding_path=data_dir / 'en_fasttext_word2vec_V100D20.json',
        split_token=' ',
    )


@pytest.fixture(scope='session')
def corpus_config(data_dir: pathlib.Path, language_config):
    return CorpusConfig(
        name='test',
        path=data_dir / 'train.txt',
        maxlen=10,
        language_config=language_config,
    )


def test_uttut_preprocessor(corpus_config):
    data_collection, _ = Preprocessor(corpus_config).preprocess()
    assert data_collection['train'].ids.shape[1] == corpus_config.maxlen


class TestWordEmbeddingCollection:

    def test_get_matrix(self):
        wordvec = WordEmbeddingCollection(
            token2index={'a': 0, 'b': 1, 'c': 2, WordEmbeddingCollection.UNK: 3},
            vectors=[[0, 1], [2, 3], [4, 5], [6, 7]],
        )
        assert np.array_equal(
            wordvec.get_matrix_of_tokens(['b', 'd is unk', 'a']),
            np.asarray([[2, 3], [6, 7], [0, 1]], dtype=np.float32),
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
