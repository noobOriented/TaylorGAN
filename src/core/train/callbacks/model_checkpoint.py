import os
import pathlib

import pydantic

from core.train.trainers import Trainer
from library.utils import format_path

from .base import Callback


class ModelCheckpoint(Callback):

    def __init__(self, args: pydantic.BaseModel, trainer: Trainer, directory: str | os.PathLike[str], period: int):
        self.args = args  # TODO global?
        self.trainer = trainer
        self.directory = pathlib.Path(directory)
        if period <= 0:
            raise ValueError("'saving_period' should be positive!")
        self.period = period

    def on_train_begin(self, is_restored: bool):
        self.directory.mkdir(exist_ok=True)
        if not is_restored:
            with open(self.directory / 'args', 'w') as f:
                f.write(self.args.model_dump_json())

    def on_epoch_end(self, epoch: int):
        if epoch % self.period == 0:
            print(f"{epoch} epochs done.")
            path = self.directory / self.checkpoint_basename(epoch)
            self.trainer.save_state(path)
            print(f"saving checkpoint to {format_path(path)}")

    def get_config(self):
        return {'directory': format_path(self.directory), 'period': self.period}

    @classmethod
    def checkpoint_basename(cls, epoch: int) -> str:
        return f'epoch{epoch}.pth'

    @classmethod
    def epoch_number(cls, path: str | os.PathLike[str]):
        return int(os.path.basename(path)[5:-4])

    @classmethod
    def latest_checkpoint(cls, directory) -> str | os.PathLike[str]:
        filename = max(
            (filename for filename in os.listdir(directory) if filename.endswith('.pth')),
            key=cls.epoch_number,
        )
        return pathlib.Path(directory, filename)
