import typing as t

from library.utils import FormatableMixin


class Callback(FormatableMixin):

    def on_train_begin(self, is_restored: bool):
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


class CallbackList(Callback):

    def __init__(self, callbacks: t.Sequence[Callback]):
        self.callbacks = callbacks

    def on_train_begin(self, is_restored: bool):
        for callback in self.callbacks:
            callback.on_train_begin(is_restored)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_batch_begin(self, batch: int):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch: int, batch_data):
        for callback in self.callbacks:
            callback.on_batch_end(batch, batch_data)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def summary(self):
        for cbk in self.callbacks:
            cbk.summary()
