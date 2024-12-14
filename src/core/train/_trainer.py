from __future__ import annotations

import os
import pathlib
import typing as t

import numpy as np
import torch

from core.models import Generator, TokenSequence
from library.utils import cache_method_call, logging_indent

from ._loss import GeneratorLoss
from ._pubsub import ListenableEvent


class GeneratorTrainer:

    def __init__(
        self,
        generator: Generator,
        optimizer: torch.optim.Optimizer,
        losses: t.Mapping[str, tuple[GeneratorLoss, float]],
    ):
        self._generator = generator
        self._generator_updater = ModuleUpdater(generator, optimizer, losses)

        self.generator_post_step_event = self._generator_updater.optimizer_post_step_event
        self.loss_update_events: dict[str, dict[str, ListenableEvent[int, float]]] = {
            self._generator.scope: self._generator_updater.loss_update_events,
        }

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=self._generator.special_tokens.EOS.idx,
            )
            with cache_method_call(self._generator, 'generate'):
                self._generator_updater.update_step(real_samples)

    def save_state(self, path: str | os.PathLike[str]):
        state_dict = [updater.state_dict() for updater in self._updaters]
        torch.save(state_dict, path)

    def load_state(self, path: str | os.PathLike[str]):
        state_dicts = torch.load(path)
        for updater, state_dict in zip(self._updaters, state_dicts):
            updater.load_state_dict(state_dict)

    def summary(self):
        for updater in self._updaters:
            trainable_params = sum(p.numel() for p in updater._module.parameters() if p.requires_grad)
            fixed_params = sum(p.numel() for p in updater._module.parameters() if not p.requires_grad)
            with logging_indent(updater._module.scope):
                with logging_indent("Model"):
                    print(f'Trainable     params: {trainable_params:>12}')
                    print(f'Non-trainable params: {fixed_params:>12,}')

                print(f'Optimizer: {updater._optimizer}')
                for loss in updater._losses.values():
                    print(loss)

    @property
    def _updaters(self) -> list[ModuleUpdater]:
        return [self._generator_updater]


class ModuleUpdater[T: torch.nn.Module, **LossP]:

    def __init__(
        self,
        module: T,
        optimizer: torch.optim.Optimizer,
        losses: t.Mapping[
            str,
            tuple[t.Callable[t.Concatenate[T, LossP], torch.Tensor], float],
        ],
    ):
        self._module = module
        self._optimizer = optimizer
        self._losses = losses

        self._step = 0
        self.optimizer_post_step_event = ListenableEvent[int]()
        self.loss_update_events = {
            name: ListenableEvent[int, float]()
            for name in losses.keys()
        }

    @property
    def step(self) -> int:
        return self._step

    def update_step(self, *args: LossP.args, **kwargs: LossP.kwargs):
        losses = {
            name: loss_fn(self._module, *args, **kwargs)
            for name, (loss_fn, _) in self._losses.items()
        }
        for name, loss_val in losses.items():
            self.loss_update_events[name](self.step, loss_val.detach().numpy())

        sum_loss: torch.Tensor = sum(self._losses[k][1] * v for k, v in losses.items())  # type: ignore
        self._optimizer.zero_grad()
        sum_loss.backward()
        self._optimizer.step()
        self._step += 1
        self.optimizer_post_step_event(self.step)

    def state_dict(self) -> dict:
        return {
            'module': self._module.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'step': self.step,
        }

    def load_state_dict(self, state_dict: dict):
        self._module.load_state_dict(state_dict['module'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._step = state_dict['step']


class ModelCheckpointSaver:

    def __init__(self, trainer: GeneratorTrainer, directory: str | os.PathLike[str]):
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
