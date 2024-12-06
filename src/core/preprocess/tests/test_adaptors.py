import numpy as np

from .._adaptors import WordEmbeddingCollection


class TestWordEmbeddingCollection:

    def test_get_matrix(self):
        wordvec = WordEmbeddingCollection(
            {'a': 0, 'b': 1, 'c': 2, WordEmbeddingCollection.UNK: 3},
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )
        assert np.array_equal(
            wordvec.get_matrix_of_tokens(['b', 'd is unk', 'a']),
            [[2, 3], [6, 7], [0, 1]],
        )
