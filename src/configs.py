from __future__ import annotations

import pathlib
import typing as t

import pydantic

from factories.data_factory import DataConfigs
from factories.generator_factory import GeneratorConfigs
from factories.trainer_factory import GANObjectiveConfigs, MLEObjectiveConfigs


class _CommonTrainingConfigs(DataConfigs, GeneratorConfigs):
    # Training
    epochs: t.Annotated[int, pydantic.Field(ge=1, description='number of training epochs.')] = 10_000
    batch_size: t.Annotated[int, pydantic.Field(ge=1, description='size of data mini-batch.')] = 64
    random_seed: t.Annotated[int | None, pydantic.Field(description='the global random seed.')] = None

    # Evaluate
    bleu: t.Annotated[
        int | None,
        pydantic.Field(ge=1, le=5, description='longest n-gram to calculate BLEU/SelfBLEU score.'),
    ] = 5
    fed: t.Annotated[
        int | None,
        pydantic.Field(description='number of sample size for FED score.'),
    ] = None

    # Save
    checkpoint_root: pathlib.Path | None = None
    serving_root: pathlib.Path | None = None
    save_period: pydantic.PositiveInt = 1

    # Logging
    tensorboard: t.Annotated[
        pathlib.Path | None,
        pydantic.Field(description='whether to log experiment on tensorboard.')
    ] = None
    tags: t.Annotated[
        list[str],
        pydantic.Field(description='additional tags to configure this training (will be used in tensorboard).'),
    ] = []

    # Dev
    profile: pathlib.Path | None = None


class MLETrainingConfigs(_CommonTrainingConfigs, MLEObjectiveConfigs, extra='forbid'):
    pass


class GANTrainingConfigs(_CommonTrainingConfigs, GANObjectiveConfigs, extra='forbid'):
    pass
