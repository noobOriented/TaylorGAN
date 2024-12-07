import os
import typing as t

import numpy as np
import numpy.typing as npt
import umsgpack


class WordEmbeddingCollection:

    UNK = '<unk>'

    def __init__(self, token2index: t.Mapping[str, int], vectors: t.Sequence[t.Sequence[float]]):
        self.token2index = token2index
        self.vectors = np.asarray(vectors, np.float32)

    @classmethod
    def load_msg(cls, path: str | os.PathLike[str]):
        with open(path, "rb") as f_in:
            params = umsgpack.unpack(f_in)
        return cls(token2index=params['token2index'], vectors=params['vector'])

    def get_matrix_of_tokens(self, token_list: t.Sequence[str]) -> npt.NDArray[np.float32]:
        return np.array([
            self._get_vector_of_token(token)
            for token in token_list
        ])

    def _get_vector_of_token(self, token: str):
        if token in self.token2index:
            return self.vectors[self.token2index[token]]
        if self.UNK in self.token2index:
            return self.vectors[self.token2index[self.UNK]]
        return np.zeros_like(self.vectors[0])
