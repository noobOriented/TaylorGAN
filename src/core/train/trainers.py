from __future__ import annotations

import abc
import os
import pathlib
import typing as t

import more_itertools
import numpy as np
import torch

from core.models import Generator
from core.models.sequence_modeling import TokenSequence
from library.utils import cache_method_call, logging_indent

from .pubsub import ListenableEvent


class Trainer(abc.ABC):

    def __init__(self, generator_updater: ModuleUpdater[Generator]):
        self.generator_updater = generator_updater
        self.generator = generator_updater.module
        self.generator_losses = generator_updater.losses

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

    def _compute_generator_loss(self, real_samples: TokenSequence) -> torch.Tensor:
        g_losses = {
            name: loss_fn(self.generator, real_samples)
            for name, (loss_fn, _) in self.generator_losses.items()
        }
        for name, loss_val in g_losses.items():
            self.generator_updater.loss_update_events[name](
                self.generator_updater.step,
                loss_val.detach().numpy(),
            )
        return sum(self.generator_losses[k][1] * v for k, v in g_losses.items())  # type: ignore


class NonParametrizedTrainer(Trainer):

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=self.generator_updater.module.special_tokens.EOS.idx,
            )
            with cache_method_call(self.generator, 'generate'):
                g_loss = self._compute_generator_loss(real_samples)
                self.generator_updater.update_step(g_loss)


class ModuleUpdater[T: torch.nn.Module]:

    def __init__(
        self,
        module: T,
        optimizer: torch.optim.Optimizer,
        losses: t.Mapping[
            str,
            tuple[t.Callable[..., torch.Tensor], float]
        ],
    ):
        self.module = module
        self.optimizer = optimizer
        self.losses = losses

        self.step = 0
        self.optimizer_post_step_event = ListenableEvent[int]()
        self.loss_update_events = {
            k: ListenableEvent[int, float]()
            for k in losses.keys()
        }

    def update_step(self, sum_loss: torch.Tensor):
        self.optimizer.zero_grad()
        sum_loss.backward()
        self.optimizer.step()
        self.optimizer_post_step_event(self.step)
        self.step += 1

    def state_dict(self) -> dict:
        return {
            'module': self.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict['module'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if op_state := state_dict['optimizer']['state']:
            self.step = more_itertools.first(op_state.values())['step']


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
