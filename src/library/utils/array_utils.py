import typing as t

import numpy as np
import numpy.typing as npt


def get_seqlens(data: np.ndarray, eos_idx: int) -> npt.NDArray[np.uint]:
    data = np.asarray(data, dtype=np.int64)
    end_mask = np.equal(data, eos_idx)
    return np.where(
        np.any(end_mask, axis=1),
        np.argmax(end_mask, axis=1),  # position of eos
        data.shape[1],  # pad length
    )


def random_sample[T](arr: t.Sequence[T], size: int) -> t.Sequence[T]:
    if size > len(arr):
        raise ValueError(f"expect `size` <= length of `arr`, Found {size} > {len(arr)}!")
    elif size == len(arr):
        return arr

    sample_ids = np.random.choice(len(arr), replace=False, size=[size])
    if isinstance(arr, np.ndarray):
        return arr[sample_ids]
    return [arr[idx] for idx in sample_ids]
