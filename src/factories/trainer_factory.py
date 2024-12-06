from __future__ import annotations

import abc
import functools
import typing as t

import pydantic
import torch
from flexparse import LookUpCall

from core.models import Discriminator, Generator
from core.objectives import MLEObjective
from core.objectives.GAN import (
    BCE, GANLossTuple, GANObjective, GumbelSoftmaxEstimator,
    ReinforceEstimator, StraightThroughEstimator, TaylorEstimator,
)
from core.objectives.regularizers import (
    EmbeddingRegularizer, EntropyRegularizer, GradientPenaltyRegularizer,
    LossScaler, SpectralRegularizer, WordVectorRegularizer,
)
from core.preprocess.record_objects import MetaData
from core.train import (
    DiscriminatorUpdater, GANTrainer, GeneratorUpdater, NonParametrizedTrainer, Trainer,
)
from core.train.optimizer import OptimizerWrapper
from factories.modules import discriminator_factory
from library.utils import ArgumentBinder


def create(args: MLEObjectiveConfigs | GANObjectiveConfigs, metadata: MetaData, generator: Generator) -> Trainer:
    creator = args.creator_cls(args, metadata, generator)
    generator_updater = GeneratorUpdater(
        generator,
        optimizer=OPTIMIZERS(args.g_optimizer)(generator.trainable_variables),
        losses=[creator.objective] + [
            _G_REGS(s)
            for s in args.g_regularizers
        ],  # TODO
    )
    return creator.create_trainer(generator_updater)


_G_REGS = LookUpCall({
    'spectral': LossScaler.as_constructor(SpectralRegularizer),
    'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
    'entropy': LossScaler.as_constructor(EntropyRegularizer),
})
OPTIMIZERS = LookUpCall(
    {
        key: ArgumentBinder(
            OptimizerWrapper.as_constructor(optim_cls),
            preserved=['params'],
        )
        for key, optim_cls in [
            ('sgd', torch.optim.SGD),
            ('rmsprop', torch.optim.RMSprop),
            ('adam', torch.optim.Adam),
        ]
    },
)


class TrainerCreator(abc.ABC):

    def __init__(self, args, meta_data, generator):
        self.args = args
        self.meta_data = meta_data
        self.generator = generator

    @abc.abstractmethod
    def create_trainer(self, generator_updater) -> Trainer:
        ...

    @property
    @abc.abstractmethod
    def objective(self) -> t.Callable:
        ...


class MLECreator(TrainerCreator):

    def create_trainer(self, generator_updater) -> NonParametrizedTrainer:
        return NonParametrizedTrainer(generator_updater)

    @property
    def objective(self):
        return MLEObjective()


class GANCreator(TrainerCreator):
    args: GANObjectiveConfigs

    def create_trainer(self, generator_updater) -> GANTrainer:
        return GANTrainer(
            generator_updater=generator_updater,
            discriminator_updater=self.create_discriminator_updater(
                self._discriminator,
                discriminator_loss=self._loss.discriminator_loss,
            ),
            d_steps=self.args.d_steps,
        )

    def create_discriminator_updater(self, discriminator, discriminator_loss):
        return DiscriminatorUpdater(
            discriminator,
            optimizer=OPTIMIZERS(self.args.d_optimizer)(discriminator.trainable_variables),
            losses=[discriminator_loss] + [_D_REGS(s) for s in self.args.d_regularizers],
        )

    @functools.cached_property
    def objective(self):
        estimator = _ESTIMATORS(self.args.estimator)
        return GANObjective(
            discriminator=self._discriminator,
            generator_loss=self._loss.generator_loss,
            estimator=estimator,
        )

    @functools.cached_property
    def _discriminator(self) -> Discriminator:
        return discriminator_factory.create(self.args, self.meta_data)

    @functools.cached_property
    def _loss(self) -> GANLossTuple:
        return {
            'alt': GANLossTuple(lambda fake_score: BCE(fake_score, labels=1.)),  # RKL - 2JS
            'JS': GANLossTuple(lambda fake_score: -BCE(fake_score, labels=0.)),  # 2JS
            'KL': GANLossTuple(lambda fake_score: -torch.exp(fake_score)),  # -sig / (1 - sig)
            'RKL': GANLossTuple(lambda fake_score: -fake_score),  # log((1 - sig) / sig)
        }[self.args.loss]


class MLEObjectiveConfigs(pydantic.BaseModel):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    g_regularizers: list[str] = []

    creator_cls: t.ClassVar = MLECreator


class GANObjectiveConfigs(pydantic.BaseModel):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    g_regularizers: list[str] = []

    discriminator: str = "cnn(activation='elu')"
    d_steps: t.Annotated[int, pydantic.Field(ge=1, description='update generator every n discriminator steps.')] = 1
    d_regularizers: list[str] = []
    d_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    d_fix_embeddings: bool = False

    loss: t.Annotated[str, pydantic.Field(description='loss function pair of GAN.')] = 'RKL'
    estimator: t.Annotated[str, pydantic.Field(description='gradient estimator for discrete sampling.')] = 'taylor'

    creator_cls: t.ClassVar = GANCreator


_ESTIMATORS = LookUpCall({
    'reinforce': ReinforceEstimator,
    'st': StraightThroughEstimator,
    'taylor': TaylorEstimator,
    'gumbel': GumbelSoftmaxEstimator,
})
_D_REGS = LookUpCall({
    'spectral': LossScaler.as_constructor(SpectralRegularizer),
    'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
    'grad_penalty': LossScaler.as_constructor(GradientPenaltyRegularizer),
    'word_vec': LossScaler.as_constructor(WordVectorRegularizer),
})
