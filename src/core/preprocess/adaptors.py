import typing as t

import numpy as np
import numpy.typing as npt
import umsgpack
from uttut.pipeline.ops import Operator
from uttut.pipeline.pipe import Pipe

from library.utils import JSONSerializableMixin, ObjectWrapper


class UttutPipeline[In, Out](JSONSerializableMixin, ObjectWrapper):

    def __init__(self, ops: t.Sequence[Operator] = ()):
        pipe = Pipe()
        for op in ops:
            pipe.add_op(op)
        self._pipe = pipe
        super().__init__(pipe)

    @classmethod
    def from_config(cls, config):
        pipe = cls()
        pipe._pipe = Pipe.deserialize(config)
        return pipe

    def get_config(self):
        return self._pipe.serialize()

    def transform_sequence(self, sequence: In) -> Out:
        return self._pipe.transform_sequence(sequence)[0]

    def summary(self):
        self._pipe.summary()
        print()


class WordEmbeddingCollection:

    UNK = '<unk>'

    def __init__(self, token2index: t.Mapping[str, int], vectors: t.Sequence[t.Sequence[float]]):
        self.token2index = token2index
        self.vectors = np.asarray(vectors, np.float32)

    @classmethod
    def load_msg(cls, path: str):
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
