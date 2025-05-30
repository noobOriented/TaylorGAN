from __future__ import annotations

import typing as t

import more_itertools
import pydantic
import torch

from core.models import Generator
from library.utils import ArgumentBinder, LookUpCall, wraps_with_new_signature
from preprocess import PreprocessResult

from ._loss import EntropyRegularizer, GeneratorLoss, mean_negative_log_likelihood
from ._trainer import GeneratorTrainer


class TrainerConfigs(pydantic.BaseModel):
    g_optimizer: OptimizerFactory = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    generator_losses: t.Annotated[
        list[str],
        pydantic.Field(alias='g_loss'),
    ] = []

    def get_trainer(self, data: PreprocessResult, generator: Generator):
        losses: dict[str, tuple[GeneratorLoss, float]] = {}
        for s in self.generator_losses:
            (reg, coeff), info = _G_REGS(s, return_info=True)
            losses[info.func_name] = (reg, coeff)

        return GeneratorTrainer(
            generator,
            optimizer=self.g_optimizer(generator.parameters()),
            losses=losses,
        )


def _concat_coeff[**P, T](
    regularizer_cls: t.Callable[P, T],
) -> t.Callable[t.Concatenate[float, P], tuple[T, float]]:

    @wraps_with_new_signature(regularizer_cls)
    def wrapper(coeff: float, *args: P.args, **kwargs: P.kwargs):
        return regularizer_cls(*args, **kwargs), coeff

    return wrapper


def _add_custom_optimizer_args(optimizer_cls: type[torch.optim.Optimizer]) -> t.Callable[..., torch.optim.Optimizer]:

    @wraps_with_new_signature(optimizer_cls)
    def factory(*args, clip_norm: float = 0, **kwargs):
        optimizer = optimizer_cls(*args, **kwargs)
        @optimizer.register_step_pre_hook
        def _(*_):
            params = more_itertools.flatten(
                g['params'] for g in optimizer.param_groups
            )
            torch.nn.utils.clip_grad_norm_(params, clip_norm)

        return optimizer

    return factory


_G_REGS = LookUpCall({
    'entropy': _concat_coeff(EntropyRegularizer),
    'NLL': lambda coeff: (mean_negative_log_likelihood, coeff),
})
_OPTIMIZERS = LookUpCall({
    key: ArgumentBinder(_add_custom_optimizer_args(optim_cls), preserved=['params'])
    for key, optim_cls in [
        ('sgd', torch.optim.SGD),
        ('rmsprop', torch.optim.RMSprop),
        ('adam', torch.optim.Adam),
    ]
})
OptimizerFactory = t.Annotated[
    t.Callable[[t.Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
    pydantic.PlainValidator(_OPTIMIZERS.parse),
]
TrainerConfigs.model_rebuild()
