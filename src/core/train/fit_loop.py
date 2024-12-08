import typing as t

import more_itertools
import numpy.typing as npt

from library.utils import batch_generator, format_highlight


class DataLoader:

    def __init__(
        self,
        dataset: npt.NDArray,
        batch_size: int,
        n_epochs: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self._callback = CustomCallback()
        self._batch = 0
        self._epoch = 1

    def skip_epochs(self, epochs: int):
        # exhaust batch_generator to make sure random state is same
        self._batch += sum(more_itertools.ilen(self._get_batch_generator()) for _ in range(epochs))
        print(f"Skip {epochs} epochs. Finish restoring process.")
        self._epoch += epochs

    def __iter__(self) -> t.Iterator[npt.NDArray]:
        self._callback.on_train_begin(self._epoch > 1)
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
        )


class CustomCallback:
    def __init__(self) -> None:
        self.on_train_begin = Event[bool]()
        self.on_epoch_begin = Event[int]()
        self.on_batch_begin = Event[int]()
        self.on_batch_end = Event[int, npt.NDArray]()
        self.on_epoch_end = Event[int]()
        self.on_train_end = Event[()]()


class Event[*T]:

    def __init__(self) -> None:
        self._hooks: list[t.Callable[[*T], t.Any]] = []

    def __call__(self, *args: *T) -> None:
        for f in self._hooks:
            f(*args)

    def attach(self, f: t.Callable[[*T], t.Any], /):
        self._hooks.append(f)
        return f
