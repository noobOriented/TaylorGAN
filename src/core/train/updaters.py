import abc
import typing as t

import torch
from more_itertools import first

from core.models.generators import Generator
from core.models.sequence_modeling import TokenSequence
from library.utils import cache_method_call, logging_indent

from .pubsub import EventHook


class ModuleUpdater[T: torch.nn.Module]:

    def __init__(
        self,
        module: T,
        optimizer: torch.optim.Optimizer,
        losses: t.Mapping[str, t.Callable[..., torch.Tensor]],
    ):
        self.module = module
        self.optimizer = optimizer
        self.losses = losses

        self.step = 0
        self.hook = EventHook[int, t.Mapping[str, float]]()

    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        ...

    def update_step(self, *args, **kwargs):
        losses = self.compute_loss(*args, **kwargs)
        sum_loss: torch.Tensor = sum(losses.values())
        loss_vals = {k: v.detach().numpy() for k, v in losses.items()}
        self.hook(self.step, loss_vals)
        self.optimizer.zero_grad()
        sum_loss.backward()
        self.optimizer.step()
        self.step += 1

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


class GeneratorUpdater(ModuleUpdater[Generator]):

    def compute_loss(self, real_samples: TokenSequence):
        with cache_method_call(self.module, 'generate'):
            return {
                name: loss(generator=self.module, real_samples=real_samples)
                for name, loss in self.losses.items()
            }
