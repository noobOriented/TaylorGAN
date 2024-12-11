import abc
import typing as t

import more_itertools
import torch

from core.models.generators import Generator
from core.models.sequence_modeling import TokenSequence
from library.utils import cache_method_call, logging_indent

from .pubsub import EventHook


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
        self.step_hook = EventHook[int]()
        self.loss_hooks = {
            k: EventHook[int, float]()
            for k in losses.keys()
        }

        @self.optimizer.register_step_post_hook
        def call_hook(*_):
            self.step_hook(self.step)
            self.step += 1

    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        ...

    def update_step(self, *args, **kwargs):
        losses = self.compute_loss(*args, **kwargs)
        for k, v in losses.items():
            self.loss_hooks[k](self.step, v.detach().numpy())

        self.optimizer.zero_grad()
        sum_loss: torch.Tensor = sum(self.losses[k][1] * v for k, v in losses.items())
        sum_loss.backward()
        self.optimizer.step()

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

    def summary(self):
        trainable_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        fixed_params = sum(p.numel() for p in self.module.parameters() if not p.requires_grad)
        with logging_indent(self.module.scope):
            with logging_indent("Model"):
                print(f'Trainable     params: {trainable_params:>12}')
                print(f'Non-trainable params: {fixed_params:>12,}')

            print(f'Optimizer: {self.optimizer}')
            with logging_indent("Objective:"):
                for loss in self.losses:
                    print(loss)


class GeneratorUpdater(ModuleUpdater[Generator]):

    def compute_loss(self, real_samples: TokenSequence):
        with cache_method_call(self.module, 'generate'):
            return {
                name: loss(generator=self.module, real_samples=real_samples)
                for name, (loss, _) in self.losses.items()
            }
