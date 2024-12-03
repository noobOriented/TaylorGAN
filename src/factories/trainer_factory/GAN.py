import functools

import torch
from flexparse import IntRange, LookUp, LookUpCall, create_action

from core.models import Discriminator
from core.objectives.GAN import (
    BCE, GANLossTuple, GANObjective, GumbelSoftmaxEstimator,
    ReinforceEstimator, StraightThroughEstimator, TaylorEstimator,
)
from core.train import DiscriminatorUpdater, GANTrainer
from factories.modules import discriminator_factory

from .trainer_factory import _OPTIMIZERS, TrainerCreator, create_optimizer_action_of


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
            optimizer=_OPTIMIZERS(self.args.d_optimizer)(discriminator.trainable_variables),
            losses=[discriminator_loss] + [
                discriminator_factory.D_REGS(s)
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

    @classmethod
    def model_args(cls):
        return discriminator_factory.MODEL_ARGS

    @classmethod
    def objective_args(cls):
        return GAN_ARGS

    @classmethod
    def regularizer_args(cls):
        return [discriminator_factory.REGULARIZER_ARG]

    @classmethod
    def optimizer_args(cls):
        return [D_OPTIMIZER_ARG]

    @functools.cached_property
    def _loss(self) -> GANLossTuple:
        return LookUp({
            'alt': GANLossTuple(lambda fake_score: BCE(fake_score, labels=1.)),  # RKL - 2JS
            'JS': GANLossTuple(lambda fake_score: -BCE(fake_score, labels=0.)),  # 2JS
            'KL': GANLossTuple(lambda fake_score: -torch.exp(fake_score)),  # -sig / (1 - sig)
            'RKL': GANLossTuple(lambda fake_score: -fake_score),  # log((1 - sig) / sig)
        })(self.args.loss)


_ESTIMATORS = LookUpCall({
    'reinforce': ReinforceEstimator,
    'st': StraightThroughEstimator,
    'taylor': TaylorEstimator,
    'gumbel': GumbelSoftmaxEstimator,
})
D_OPTIMIZER_ARG = create_optimizer_action_of('discriminator')
GAN_ARGS = [
    create_action(
        '--loss',
        default='RKL',
        help='loss function pair of GAN.',
    ),
    create_action(
        '--estimator',
        default='taylor',
        help='\n'.join([
            'gradient estimator for discrete sampling.',
            'custom options and registry: ',
            *_ESTIMATORS.get_helps(),
        ]) + "\n",
    ),
    create_action(
        '--d-steps',
        type=IntRange(minval=1),
        default=1,
        help='update generator every n discriminator steps.',
    ),
]
