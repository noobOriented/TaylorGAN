import os
import typing as t

import numpy as np
import numpy.typing as npt
import torch

from core.models import AutoRegressiveGenerator, Generator
from library.utils import batch_generator
from preprocess import Tokenizer


class TextGenerator:
    '''Facade class of Generator model'''

    BATCH_SIZE = 64

    def __init__(self, generator: Generator, tokenizer: Tokenizer):
        self.generator = generator
        self._tokenizer = tokenizer

    def generate_texts(self, size: int, *, temperature: float = 1) -> list[str]:
        return [
            self._tokenizer.ids_to_text(ids)
            for ids in self.generate_ids(size, temperature=temperature)
        ]

    def generate_ids(self, size: int, *, temperature: float = 1) -> npt.NDArray[np.uint]:
        return np.concatenate(
            [
                self.generator.forward(
                    torch.tensor(batch_size),
                    torch.tensor(self._tokenizer.maxlen),
                    temperature=torch.tensor(temperature),
                )
                for batch_size in compute_batch_size(size, self.BATCH_SIZE)
            ],
            axis=0,
        )

    def perplexity(self, inputs: np.ndarray) -> float:
        total_NLL = total_words = 0.
        generator = t.cast(AutoRegressiveGenerator, self.generator)
        with torch.no_grad():
            for batch_array in batch_generator(inputs, self.BATCH_SIZE):
                batch_tensor = torch.from_numpy(batch_array)
                batch_NLL = generator.seq_neg_logprobs(batch_tensor)
                total_NLL += batch_NLL.sum()
                # TODO seqlen
                total_words += inputs.shape[0] * inputs.shape[1]

        avg_NLL: torch.Tensor = total_NLL / total_words
        return avg_NLL.exp().numpy()

    def export_traced(self):
        inputs = {
            'forward': (
                torch.tensor(1),
                torch.tensor(self._tokenizer.maxlen),
                torch.tensor(1.),
            ),
            'seq_neg_logprobs': torch.zeros([1, self._tokenizer.maxlen], dtype=torch.int64),
        }
        return torch.jit.trace_module(self.generator, inputs)

    def ids_to_text(self, word_ids):
        return self._tokenizer.ids_to_text(word_ids)

    @classmethod
    def load_traced(cls, path: str | os.PathLike[str], tokenizer: Tokenizer):
        return cls(torch.jit.load(str(path)), tokenizer)


def compute_batch_size(total_size: int, batch_size: int) -> t.Iterator[int]:
    q, m = divmod(total_size, batch_size)
    for _ in range(q):
        yield batch_size
    if m:
        yield m
