import abc
import typing as t

import torch


class Regularizer(t.Protocol):

    @abc.abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        ...
