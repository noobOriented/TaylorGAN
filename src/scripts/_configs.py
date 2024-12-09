from __future__ import annotations

import typing as t

import pydantic

from factories.callback_factory import CallbackConfigs
from factories.data_factory import DataConfigs
from factories.generator_factory import GeneratorConfigs
from factories.trainer_factory import GANObjectiveConfigs, MLEObjectiveConfigs


class _CommonTrainingConfigs(CallbackConfigs, DataConfigs, GeneratorConfigs):
    epochs: t.Annotated[int, pydantic.Field(ge=1, description='number of training epochs.')] = 10_000
    batch_size: t.Annotated[int, pydantic.Field(ge=1, description='size of data mini-batch.')] = 64
    random_seed: t.Annotated[int | None, pydantic.Field(description='the global random seed.')] = None


class MLETrainingConfigs(_CommonTrainingConfigs, MLEObjectiveConfigs, extra='forbid'):
    pass


class GANTrainingConfigs(_CommonTrainingConfigs, GANObjectiveConfigs, extra='forbid'):
    pass
