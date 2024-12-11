import typing as t

import torch


class LossCollection:

    def __init__(self, total: torch.Tensor, **observables):
        self.total = total
        self.observables = observables

    def __radd__(self, other) -> t.Self:
        return self + other

    def __add__(self, other) -> t.Self:
        if isinstance(other, LossCollection):
            return self.__class__(
                self.total + other.total,
                **self.observables,
                **other.observables,
            )
        if other == 0:
            return self.__class__(self.total + 0, **self.observables)
        return NotImplemented
