from contextlib import contextmanager

import numpy as np
from tqdm import tqdm
import typing as t
from .file_helper import count_lines


@contextmanager
def tqdm_open(filepath, mode='r'):
    total = count_lines(filepath)
    with open(filepath, mode) as f:
        yield tqdm(f, total=total, unit='line')


def batch_generator[S: t.Sequence | np.ndarray](
    data: S, batch_size: int, *,
    shuffle: bool = False, full_batch_only: bool = False,
) -> t.Iterator[S]:
    total = len(data)
    if full_batch_only:
        if total < batch_size:
            raise ValueError
        stop = total - batch_size + 1
    else:
        stop = total

    if shuffle:
        ids = np.random.permutation(total)
        for start in range(0, stop, batch_size):
            batch_ids = ids[start: start + batch_size]
            yield data[batch_ids]
    else:
        for start in range(0, stop, batch_size):
            yield data[start: start + batch_size]
