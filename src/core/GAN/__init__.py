from ._trainer_factory import GANObjectiveConfigs
from .discriminators import Discriminator, GradientPenaltyRegularizer, WordVectorRegularizer
from .loss import (
    BCE, GANLossTuple, GANObjective, GumbelSoftmaxEstimator, ReinforceEstimator,
    SoftmaxEstimator, StraightThroughEstimator, TaylorEstimator,
)
from .trainer import GANTrainer
