from __future__ import annotations

import functools
import typing as t

import pydantic
import torch
from flexparse import LookUpCall

from core.models import Generator
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


class MLEObjectiveConfigs(pydantic.BaseModel):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    g_regularizers: list[str] = []

    def create_trainer(self, metadata: MetaData, generator: Generator):
        generator_updater = GeneratorUpdater(
            generator,
            optimizer=_OPTIMIZERS(self.g_optimizer)(generator.trainable_variables),
            losses=[MLEObjective()] + [_G_REGS(s) for s in self.g_regularizers],
        )
        return NonParametrizedTrainer(generator_updater)


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

    def create_trainer(self, metadata: MetaData, generator: Generator):
        discriminator = discriminator_factory.create(self, metadata)
        estimator = _ESTIMATORS(self.estimator)
        objective = GANObjective(
            discriminator=discriminator,
            generator_loss=self._loss_tuple.generator_loss,
            estimator=estimator,
        )
        generator_updater = GeneratorUpdater(
            generator,
            optimizer=_OPTIMIZERS(self.g_optimizer)(generator.trainable_variables),
            losses=[objective] + [_G_REGS(s) for s in self.g_regularizers],
        )
        return GANTrainer(
            generator_updater=generator_updater,
            discriminator_updater=DiscriminatorUpdater(
                discriminator,
                optimizer=_OPTIMIZERS(self.d_optimizer)(discriminator.trainable_variables),
                losses=[self._loss_tuple.discriminator_loss] + [_D_REGS(s) for s in self.d_regularizers],
            ),
            d_steps=self.d_steps,
        )

    @functools.cached_property
    def _loss_tuple(self) -> GANLossTuple:
        return {
            'alt': GANLossTuple(lambda fake_score: BCE(fake_score, labels=1.)),  # RKL - 2JS
            'JS': GANLossTuple(lambda fake_score: -BCE(fake_score, labels=0.)),  # 2JS
            'KL': GANLossTuple(lambda fake_score: -torch.exp(fake_score)),  # -sig / (1 - sig)
            'RKL': GANLossTuple(lambda fake_score: -fake_score),  # log((1 - sig) / sig)
        }[self.loss]



_G_REGS = LookUpCall({
    'spectral': LossScaler.as_constructor(SpectralRegularizer),
    'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
    'entropy': LossScaler.as_constructor(EntropyRegularizer),
})
_OPTIMIZERS = LookUpCall(
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
