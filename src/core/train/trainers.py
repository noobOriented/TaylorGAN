import abc
import os
import pathlib
import typing as t

import numpy as np
import torch

from core.models.sequence_modeling import TokenSequence
from library.utils import logging_indent

from .updaters import GeneratorUpdater, ModuleUpdater


class Trainer(abc.ABC):

    def __init__(self, generator_updater: GeneratorUpdater):
        self.generator_updater = generator_updater

    @abc.abstractmethod
    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        ...

    def save_state(self, path: str | os.PathLike[str]):
        state_dict = [updater.state_dict() for updater in self.updaters]
        torch.save(state_dict, path)

    def load_state(self, path: str | os.PathLike[str]):
        state_dicts = torch.load(path)
        for updater, state_dict in zip(self.updaters, state_dicts):
            updater.load_state_dict(state_dict)

    @property
    def updaters(self) -> list[ModuleUpdater]:
        return [self.generator_updater]

    def summary(self):
        for updater in self.updaters:
            trainable_params = sum(p.numel() for p in updater.module.parameters() if p.requires_grad)
            fixed_params = sum(p.numel() for p in updater.module.parameters() if not p.requires_grad)
            with logging_indent(updater.module.scope):
                with logging_indent("Model"):
                    print(f'Trainable     params: {trainable_params:>12}')
                    print(f'Non-trainable params: {fixed_params:>12,}')

                print(f'Optimizer: {updater.optimizer}')
                with logging_indent("Objective:"):
                    for loss in updater.losses:
                        print(loss)


class NonParametrizedTrainer(Trainer):

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=self.generator_updater.module.special_tokens.EOS.idx,
            )
            self.generator_updater.update_step(real_samples)


class ModelCheckpointSaver:

    def __init__(self, trainer: Trainer, directory: str | os.PathLike[str]):
        self.trainer = trainer
        self.directory = pathlib.Path(directory)

    def save(self, epoch: int):
        path = self.directory / self.checkpoint_basename(epoch)
        self.trainer.save_state(path)
        print(f"saving checkpoint to {path}")

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
