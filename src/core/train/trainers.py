import abc
import os
import pathlib
import typing as t

import numpy as np
import torch

from core.models.sequence_modeling import TokenSequence
from library.utils import format_path

from .updaters import DiscriminatorUpdater, GeneratorUpdater


class Trainer(abc.ABC):

    def __init__(self, generator_updater: GeneratorUpdater):
        self.generator_updater = generator_updater

    @abc.abstractmethod
    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        pass

    def save_state(self, path: str | os.PathLike[str]):
        state_dict = [updater.state_dict() for updater in self.updaters]
        torch.save(state_dict, path)

    def load_state(self, path: str | os.PathLike[str]):
        state_dicts = torch.load(path)
        for updater, state_dict in zip(self.updaters, state_dicts):
            updater.load_state_dict(state_dict)

    @property
    def updaters(self):
        return [self.generator_updater]

    def summary(self):
        for updater in self.updaters:
            updater.summary()


class NonParametrizedTrainer(Trainer):

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=1,
            )
            self.generator_updater.update_step(real_samples)


class GANTrainer(Trainer):

    def __init__(
        self,
        generator_updater: GeneratorUpdater,
        discriminator_updater: DiscriminatorUpdater,
        d_steps: int = 1,
    ):
        super().__init__(generator_updater)
        self.discriminator_updater = discriminator_updater
        self.d_steps = d_steps

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=1,
            )
            self.discriminator_updater.update_step(
                real_samples=real_samples,
                fake_samples=self.generator_updater.module.generate(*batch_data.shape),
            )
            if self.discriminator_updater.step % self.d_steps == 0:
                self.generator_updater.update_step(real_samples)

    @property
    def updaters(self):
        return super().updaters + [self.discriminator_updater]


class ModelCheckpointSaver:

    def __init__(self, trainer: Trainer, directory: str | os.PathLike[str]):
        self.trainer = trainer
        self.directory = pathlib.Path(directory)

    def save(self, epoch: int):
        path = self.directory / self.checkpoint_basename(epoch)
        self.trainer.save_state(path)
        print(f"saving checkpoint to {format_path(path)}")

    @classmethod
    def checkpoint_basename(cls, epoch: int) -> str:
        return f'epoch{epoch}.pth'

    @classmethod
    def epoch_number(cls, path: str | os.PathLike[str]):
        return int(os.path.basename(path)[5:-4])

    @classmethod
    def latest_checkpoint(cls, directory: str | os.PathLike[str]) -> str | os.PathLike[str]:
        filename = max(
            (filename for filename in os.listdir(directory) if filename.endswith('.pth')),
            key=cls.epoch_number,
        )
        return pathlib.Path(directory, filename)
