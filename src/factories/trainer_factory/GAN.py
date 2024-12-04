import functools

import torch
from flexparse import LookUpCall

from core.models import Discriminator
from core.objectives.GAN import (
    BCE, GANLossTuple, GANObjective, GumbelSoftmaxEstimator,
    ReinforceEstimator, StraightThroughEstimator, TaylorEstimator,
)
from core.objectives.regularizers import (
    EmbeddingRegularizer, GradientPenaltyRegularizer,
    LossScaler, SpectralRegularizer, WordVectorRegularizer,
)
from core.train import DiscriminatorUpdater, GANTrainer
from factories.modules import discriminator_factory

from .trainer_factory import OPTIMIZERS, TrainerCreator


class GANCreator(TrainerCreator):

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
            losses=[discriminator_loss] + [
                _D_REGS(s)
                for s in self.args.d_regularizers
            ],
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
