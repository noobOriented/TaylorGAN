from .. import Preprocessor


def test_uttut_preprocessor(corpus_config):
    data_collection, _ = Preprocessor(corpus_config).preprocess()
    assert data_collection['train'].ids.shape[1] == corpus_config.maxlen
