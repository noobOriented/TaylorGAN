import typing as t

import more_itertools
import numpy.typing as npt

from library.utils import batch_generator, format_highlight

from .callbacks import Callback, NullCallback


class DataLoader:

    def __init__(
        self,
        dataset: npt.NDArray,
        batch_size: int,
        n_epochs: int,
        callback: Callback | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self._callback = callback or NullCallback()
        self._batch = 0
        self._epoch = 1

    def skip_epochs(self, epochs: int):
        # exhaust batch_generator to make sure random state is same
        self._batch += sum(more_itertools.ilen(self._get_batch_generator()) for _ in range(epochs))
        print(f"Skip {epochs} epochs. Finish restoring process.")
        self._epoch += epochs

    def __iter__(self) -> t.Iterator[npt.NDArray]:
        self._callback.on_train_begin(is_restored=self._epoch > 1)
        print(format_highlight("Start Training"))
        while self._epoch <= self.n_epochs:
            self._callback.on_epoch_begin(self._epoch)
            for batch_data in self._get_batch_generator():
                self._callback.on_batch_begin(self._batch)
                yield batch_data
                self._batch += 1
                self._callback.on_batch_end(self._batch, batch_data)

            self._callback.on_epoch_end(self._epoch)
            self._epoch += 1

        self._callback.on_train_end()

    def _get_batch_generator(self):
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            full_batch_only=True,
        )
