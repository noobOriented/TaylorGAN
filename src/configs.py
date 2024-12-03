from __future__ import annotations

import pathlib
import typing as t

import pydantic
from factories.trainer_factory.GAN import GANCreator
from factories.trainer_factory.MLE import MLECreator


class _BaseModelExtraForbid(pydantic.BaseModel, extra='forbid'):
    pass


class DataConfigs(_BaseModelExtraForbid):
    dataset: t.Annotated[str, pydantic.Field(description='the choice of corpus.')]
    maxlen: t.Annotated[int | None, pydantic.Field(ge=1, description='the max length of sequence padding.')] = None
    vocab_size: t.Annotated[
        int | None,
        pydantic.Field(ge=1, description='the maximum number of tokens. ordered by descending frequency.'),
    ] = None


class ModelConfigs(_BaseModelExtraForbid):
    generator: str = 'gru'
    tie_embeddings: t.Annotated[
        bool,
        pydantic.Field(description="whether to tie the weights of generator's input/presoftmax embeddings."),
    ] = False
    g_fix_embeddings: bool = False


class GANModelConfigs(ModelConfigs):
    discriminator: str = "cnn(activation='elu')"
    d_fix_embeddings: bool = False


class ObjectiveConfigs(_BaseModelExtraForbid):
    g_regularizers: list[str] = []


class GANObjectiveConfigs(_BaseModelExtraForbid):
    loss: t.Annotated[str, pydantic.Field(description='loss function pair of GAN.')] = 'RKL'
    estimator: t.Annotated[str, pydantic.Field(description='gradient estimator for discrete sampling.')] = 'taylor'
    d_steps: t.Annotated[int, pydantic.Field(ge=1, description='update generator every n discriminator steps.')] = 1
    g_regularizers: list[str] = []
    d_regularizers: list[str] = []


class OptimizerConfigs(_BaseModelExtraForbid):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'


class GANOptimizerConfigs(_BaseModelExtraForbid):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    d_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'


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


class MLETrainingConfigs(_CommonTrainingConfigs, ModelConfigs, ObjectiveConfigs, OptimizerConfigs):
    creator_cls: t.ClassVar = MLECreator


class GANTrainingConfigs(_CommonTrainingConfigs, GANModelConfigs, GANObjectiveConfigs, GANOptimizerConfigs):
    creator_cls: t.ClassVar = GANCreator
