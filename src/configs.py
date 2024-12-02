from __future__ import annotations

import pathlib
import typing as t

import pydantic


class _BaseModelExtraForbid(pydantic.BaseModel, extra='forbid'):
    pass


class _CommonTrainingConfigs(_BaseModelExtraForbid):
    data: DataConfigs
    train: TrainConfigs
    evaluate: EvaluateConfigs
    save: SaveConfigs
    logging: LoggingConfigs


class MLETrainingConfigs(_CommonTrainingConfigs):
    model: ModelConfigs
    objective: ObjectiveConfigs
    optimizer: OptimizerConfigs


class GANTrainingConfigs(_CommonTrainingConfigs):
    model: GANModelConfigs
    objective: GANObjectiveConfigs
    optimizer: GANOptimizerConfigs


class DataConfigs(_BaseModelExtraForbid):
    dataset: str
    maxlen: pydantic.PositiveInt | None = None
    vocab_size: pydantic.PositiveInt | None = None


class ModelConfigs(_BaseModelExtraForbid):
    generator: str  # TODO args
    tie_embeddings: bool = False
    fix_generator_embeddings: bool = False


class GANModelConfigs(ModelConfigs):
    discriminator: str  # TODO args
    fix_discriminator_embeddings: bool = False


class ObjectiveConfigs(_BaseModelExtraForbid):
    generator_regularizers: list = []


class GANObjectiveConfigs(_BaseModelExtraForbid):
    loss: str = 'RKL'  # TODO args
    estimator: str = 'taylor'  # TODO args
    discriminator_steps: pydantic.PositiveInt = 1
    generator_regularizers: list = []
    discriminator_regularizers: list = []


class OptimizerConfigs(_BaseModelExtraForbid):
    generator_optimizer: str


class GANOptimizerConfigs(_BaseModelExtraForbid):
    generator_optimizer: str
    discriminator_optimizer: str


class TrainConfigs(_BaseModelExtraForbid):
    epochs: pydantic.PositiveInt = 10_000
    batch_size: pydantic.PositiveInt = 64
    random_seed: int | None = None


class EvaluateConfigs(_BaseModelExtraForbid):
    bleu: t.Annotated[int, pydantic.Field(ge=1, le=5)] = 5
    fed: pydantic.PositiveInt | None = None


class SaveConfigs(_BaseModelExtraForbid):
    checkpoint: pathlib.Path
    serving: pathlib.Path
    period: pydantic.PositiveInt = 1


class LoggingConfigs(_BaseModelExtraForbid):
    tensorboard: pathlib.Path
    tags: list[str] = []
