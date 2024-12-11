import abc
import typing as t

import torch
from more_itertools import first

from core.objectives.collections import LossCollection
from library.utils import cache_method_call, logging_indent

from .pubsub import EventHook


class ModuleUpdater:

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        losses: t.Sequence[t.Callable[..., LossCollection]],
    ):
        self.module = module
        self.optimizer = optimizer
        self.losses = losses

        self.step = 0
        self.hook = EventHook[int, t.Mapping[str, float]]()

    @abc.abstractmethod
    def update_step(self):
        pass

    def state_dict(self) -> dict:
        return {
            'module': self.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict['module'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if state_dict['optimizer']['state']:
            self.step = first(state_dict['optimizer']['state'].values())['step']

    @property
    def info(self):
        return f"{self.module.scope[0]} {str(self.module).splitlines()[0]}"  # FIXME

    def summary(self):
        with logging_indent(self.module.scope):
            with logging_indent("Model"):
                print(f'Trainable     params: {_count_numel(self.module.trainable_variables):>12}')
                print(f'Non-trainable params: {_count_numel(self.module.non_trainable_variables):>12,}')

            print(f'Optimizer: {self.optimizer}')
            with logging_indent("Objective:"):
                for loss in self.losses:
                    print(loss)


def _count_numel(params) -> int:
    return sum(p.numel() for p in params)


class GeneratorUpdater(ModuleUpdater):

    def update_step(self, real_samples):
        with cache_method_call(self.module, 'generate'):
            loss_collection: LossCollection = sum(
                loss(generator=self.module, real_samples=real_samples)
                for loss in self.losses
            )  # type: ignore

        self.step += 1
        self.optimizer.zero_grad()
        loss_collection.total.backward()
        losses = {
            key: tensor.detach().numpy()
            for key, tensor in loss_collection.observables.items()
        }
        self.hook(self.step, losses)
        self.optimizer.step()
