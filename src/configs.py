from __future__ import annotations

import pathlib
import typing as t

import pydantic

from factories.data_factory import DataConfigs
from factories.modules.generator import GeneratorConfigs
from factories.trainer_factory import GANObjectiveConfigs, MLEObjectiveConfigs


class _BaseModelExtraForbid(pydantic.BaseModel, extra='forbid'):
    pass


class TrainConfigs(_BaseModelExtraForbid):
    epochs: t.Annotated[int, pydantic.Field(ge=1, description='number of training epochs.')] = 10_000
    batch_size: t.Annotated[int, pydantic.Field(ge=1, description='size of data mini-batch.')] = 64
    random_seed: t.Annotated[int | None, pydantic.Field(description='the global random seed.')] = None


class EvaluateConfigs(_BaseModelExtraForbid):
    bleu: t.Annotated[
        int,
        pydantic.Field(ge=1, le=5, description='longest n-gram to calculate BLEU/SelfBLEU score.'),
    ] = 5
    fed: t.Annotated[
        int | None,
        pydantic.Field(description='number of sample size for FED score.'),
    ] = None


class SaveConfigs(_BaseModelExtraForbid):
    checkpoint_root: pathlib.Path | None = None
    serving_root: pathlib.Path | None = None
    save_period: pydantic.PositiveInt = 1


class LoggingConfigs(_BaseModelExtraForbid):
    tensorboard: t.Annotated[
        pathlib.Path | None,
        pydantic.Field(description='whether to log experiment on tensorboard.')
    ] = None
    tags: t.Annotated[
        list[str],
        pydantic.Field(description='additional tags to configure this training (will be used in tensorboard).'),
    ] = []


class _CommonTrainingConfigs(DataConfigs, TrainConfigs, EvaluateConfigs, SaveConfigs, LoggingConfigs):
    profile: pathlib.Path | None = None


class MLETrainingConfigs(_CommonTrainingConfigs, GeneratorConfigs, MLEObjectiveConfigs):
    pass


class GANTrainingConfigs(_CommonTrainingConfigs, GeneratorConfigs, GANObjectiveConfigs):
    pass
