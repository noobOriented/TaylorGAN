import abc
import typing as t

from library.utils import ObjectWrapper, format_object, wraps_with_new_signature

from ..collections import LossCollection


class Regularizer(t.Protocol):

    @abc.abstractmethod
    def __call__(self, **kwargs) -> LossCollection:
        ...


class LossScaler(ObjectWrapper[Regularizer]):

    def __init__(self, regularizer: Regularizer, coeff: float):
        super().__init__(regularizer)
        self.regularizer = regularizer
        self.coeff = coeff

    def __call__(self, **kwargs) -> LossCollection:
        loss = self.regularizer(**kwargs)
        return LossCollection(self.coeff * loss.total, **loss.observables)

    @classmethod
    def as_constructor(cls, regularizer_cls: t.Callable[..., Regularizer]) -> t.Callable[..., Regularizer]:

        @wraps_with_new_signature(regularizer_cls)
        def wrapper(coeff: float, *args, **kwargs):
            regularizer = regularizer_cls(*args, **kwargs)
            return cls(regularizer, coeff=coeff)

        return wrapper

    def __str__(self):
        return format_object(self.regularizer, coeff=self.coeff, **self.regularizer.__dict__)
