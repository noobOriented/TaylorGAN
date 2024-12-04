from __future__ import annotations

import abc
import typing as t

import torch
from flexparse import LookUpCall

from core.models import Generator
from core.objectives.regularizers import (
    EmbeddingRegularizer, EntropyRegularizer, LossScaler, SpectralRegularizer,
)
from core.preprocess.record_objects import MetaData
from core.train import GeneratorUpdater, Trainer
from core.train.optimizer import OptimizerWrapper
from library.utils import ArgumentBinder


def create(args, metadata: MetaData, generator: Generator) -> Trainer:
    creator: TrainerCreator = args.creator_cls(args, metadata, generator)
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
