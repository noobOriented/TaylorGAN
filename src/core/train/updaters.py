import typing as t

import more_itertools
import torch

from core.models.generators import Generator

from .pubsub import Event


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
        self.optimizer_post_step_event = Event[int]()
        self.loss_update_events = {
            k: Event[int, float]()
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


class GeneratorUpdater(ModuleUpdater[Generator]):
    pass
