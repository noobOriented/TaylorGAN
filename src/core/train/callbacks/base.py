import typing as t

import numpy.typing as npt

from core.train.pubsub import Event
from library.utils import FormatableMixin


class Callback(FormatableMixin):

    def on_train_begin(self, is_restored: bool, /):
        pass

    def on_epoch_begin(self, epoch: int, /):
        pass

    def on_batch_begin(self, batch: int, /):
        pass

    def on_batch_end(self, batch: int, batch_data, /):
        pass

    def on_epoch_end(self, epoch: int, /):
        pass

    def on_train_end(self):
        pass

    def get_config(self):
        return {}
    
    def summary(self):
        print(self)


NullCallback = Callback


class CustomCallback:
    def __init__(self) -> None:
        self.on_train_begin = Event[bool]()
        self.on_epoch_begin = Event[int]()
        self.on_batch_begin = Event[int]()
        self.on_batch_end = Event[int, npt.NDArray]()
        self.on_epoch_end = Event[int]()
        self.on_train_end = Event[()]()
