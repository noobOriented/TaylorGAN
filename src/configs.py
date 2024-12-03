from __future__ import annotations

import pathlib
import typing as t

import pydantic
from factories.trainer_factory.GAN import GANCreator
from factories.trainer_factory.MLE import MLECreator


class _BaseModelExtraForbid(pydantic.BaseModel, extra='forbid'):
    pass


class DataConfigs(_BaseModelExtraForbid):
    dataset: str
    maxlen: pydantic.PositiveInt | None = None
    vocab_size: pydantic.PositiveInt | None = None


class ModelConfigs(_BaseModelExtraForbid):
    generator: str = 'gru'
    tie_embeddings: bool = False
    g_fix_embeddings: bool = False


class GANModelConfigs(ModelConfigs):
    discriminator: str = "cnn(activation='elu')"
    d_fix_embeddings: bool = False


class ObjectiveConfigs(_BaseModelExtraForbid):
    g_regularizers: list = []


class GANObjectiveConfigs(_BaseModelExtraForbid):
    loss: str = 'RKL'  # TODO args
    estimator: str = 'taylor'  # TODO args
    d_steps: pydantic.PositiveInt = 1
    g_regularizers: list = []
    d_regularizers: list = []


class OptimizerConfigs(_BaseModelExtraForbid):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'


class GANOptimizerConfigs(_BaseModelExtraForbid):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    d_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'


class TrainConfigs(_BaseModelExtraForbid):
    epochs: pydantic.PositiveInt = 10_000
    batch_size: pydantic.PositiveInt = 64
    random_seed: int | None = None


class EvaluateConfigs(_BaseModelExtraForbid):
    bleu: t.Annotated[int, pydantic.Field(ge=1, le=5)] = 5
    fed: pydantic.PositiveInt | None = None


class SaveConfigs(_BaseModelExtraForbid):
    checkpoint_root: pathlib.Path | None = None
    serving_root: pathlib.Path | None = None
    save_period: pydantic.PositiveInt = 1


class LoggingConfigs(_BaseModelExtraForbid):
    tensorboard: pathlib.Path | None = None
    tags: list[str] = []


class _CommonTrainingConfigs(DataConfigs, TrainConfigs, EvaluateConfigs, SaveConfigs, LoggingConfigs):
    profile: pathlib.Path | None = None


class MLETrainingConfigs(_CommonTrainingConfigs, ModelConfigs, ObjectiveConfigs, OptimizerConfigs):
    creator_cls: t.ClassVar = MLECreator


class GANTrainingConfigs(_CommonTrainingConfigs, GANModelConfigs, GANObjectiveConfigs, GANOptimizerConfigs):
    creator_cls: t.ClassVar = GANCreator
