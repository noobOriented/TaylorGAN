import abc
import typing as t

import torch

from library.utils import ObjectWrapper, format_object, wraps_with_new_signature

from ..collections import LossCollection


class Regularizer(t.Protocol):
    loss_name: str

    @abc.abstractmethod
    def __call__(self, **kwargs) -> LossCollection:
        ...


class LossScaler(ObjectWrapper):

    def __init__(self, regularizer: Regularizer, coeff: float):
        super().__init__(regularizer)
        self.regularizer = regularizer
        self.coeff = coeff

    def __call__(self, **kwargs):
        loss = self.regularizer(**kwargs)
        if isinstance(loss, LossCollection):
            return loss
        
        observables = {self.regularizer.loss_name: loss}
        return LossCollection(self.coeff * loss, **observables)

    @classmethod
    def as_constructor(cls, regularizer_cls):

        @wraps_with_new_signature(regularizer_cls)
        def wrapper(coeff, *args, **kwargs):
            return cls(
                regularizer=regularizer_cls(*args, **kwargs),
                coeff=coeff,
            )

        return wrapper

    def __str__(self):
        params = {'coeff': self.coeff, **self.regularizer.__dict__}
        return format_object(self.regularizer, **params)
