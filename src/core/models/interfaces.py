import abc
import typing as t

import more_itertools
import torch


class ModuleInterface(t.Protocol):

    def parameters(self):
        return more_itertools.flatten(network.parameters() for network in self.networks)

    def modules(self):
        return more_itertools.flatten(network.modules() for network in self.networks)

    @property
    def trainable_variables(self):
        return [param for param in self.parameters() if param.requires_grad]

    @property
    def non_trainable_variables(self):
        return [param for param in self.parameters() if not param.requires_grad]

    @property
    @abc.abstractmethod
    def networks(self) -> list[torch.nn.Module]:
        ...
