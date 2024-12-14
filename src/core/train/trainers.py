from __future__ import annotations

import os
import pathlib
import typing as t

import numpy as np
import torch

from core.losses import GeneratorLoss
from core.models import Generator
from core.models.sequence_modeling import TokenSequence
from library.utils import cache_method_call, logging_indent

from .pubsub import ListenableEvent


class GeneratorTrainer:

    def __init__(
        self,
        generator: Generator,
        optimizer: torch.optim.Optimizer,
        losses: t.Mapping[str, tuple[GeneratorLoss, float]],
    ):
        self._generator = generator
        self._losses = losses
        self._generator_state = ModuleTrainingState(generator, optimizer)

        self.generator_post_step_event = self._generator_state.optimizer_post_step_event
        self.loss_update_events: dict[str, dict[str, ListenableEvent[int, float]]] = {
            self._generator.scope: {
                k: ListenableEvent[int, float]()
                for k in self._losses.keys()
            },
        }

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=self._generator.special_tokens.EOS.idx,
            )
            with cache_method_call(self._generator, 'generate'):
                g_loss = self._compute_generator_loss(real_samples)
                self._generator_state.update_step(g_loss)

    def save_state(self, path: str | os.PathLike[str]):
        state_dict = [updater.state_dict() for updater in self._module_states]
        torch.save(state_dict, path)

    def load_state(self, path: str | os.PathLike[str]):
        state_dicts = torch.load(path)
        for updater, state_dict in zip(self._module_states, state_dicts):
            updater.load_state_dict(state_dict)

    def summary(self):
        for updater in self._module_states:
            trainable_params = sum(p.numel() for p in updater._module.parameters() if p.requires_grad)
            fixed_params = sum(p.numel() for p in updater._module.parameters() if not p.requires_grad)
            with logging_indent(updater._module.scope):
                with logging_indent("Model"):
                    print(f'Trainable     params: {trainable_params:>12}')
                    print(f'Non-trainable params: {fixed_params:>12,}')

                print(f'Optimizer: {updater._optimizer}')
            # TODO losses

    @property
    def _module_states(self) -> list[ModuleTrainingState]:
        return [self._generator_state]

    def _compute_generator_loss(self, real_samples: TokenSequence) -> torch.Tensor:
        g_losses = {
            name: loss_fn(self._generator, real_samples)
            for name, (loss_fn, _) in self._losses.items()
        }
        for name, loss_val in g_losses.items():
            self.loss_update_events[self._generator.scope][name](
                self._generator_state.step,
                loss_val.detach().numpy(),
            )
        return sum(self._losses[k][1] * v for k, v in g_losses.items())  # type: ignore


class ModuleTrainingState[T: torch.nn.Module]:

    def __init__(self, module: T, optimizer: torch.optim.Optimizer):
        self._module = module
        self._optimizer = optimizer
        self._step = 0
        self.optimizer_post_step_event = ListenableEvent[int]()

    @property
    def step(self) -> int:
        return self._step

    def update_step(self, sum_loss: torch.Tensor):
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
