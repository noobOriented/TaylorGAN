from __future__ import annotations

import abc
import typing as t

import torch
from flexparse import Action, ArgumentParser, LookUpCall

from core.models import Generator
from core.train import GeneratorUpdater, Trainer
from core.train.optimizer import OptimizerWrapper
from factories.modules import generator_factory
from library.utils import ArgumentBinder

from ..utils import create_factory_action


def create(args, meta_data, generator: Generator) -> Trainer:
    creator = args.creator_cls(args, meta_data, generator)
    generator_updater = GeneratorUpdater(
        generator,
        optimizer=args.g_optimizer(generator.trainable_variables),
        losses=[creator.objective] + args.g_regularizers,
    )
    return creator.create_trainer(generator_updater)


def create_optimizer_action_of(module_name: str):
    return create_factory_action(
        f'--{module_name[0]}-optimizer',
        type=LookUpCall(
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
        ),
        default='adam(lr=1e-4, betas=(0.5, 0.999), clip_norm=10)',
        help_prefix=f"{module_name}'s optimizer.\n",
    )


G_OPTIMIZER_ARG = create_optimizer_action_of('generator')


def create_parser(algorithm: type[TrainerCreator]):
    parser = ArgumentParser(add_help=False)
    parser.add_argument_group(
        'model',
        description="Model's structure & hyperparameters.",
        actions=[
            *generator_factory.MODEL_ARGS,
            *algorithm.model_args(),
        ],
    )
    parser.add_argument_group(
        'objective',
        description="Model's objective.",
        actions=[
            *algorithm.objective_args(),
            generator_factory.REGULARIZER_ARG,
            *algorithm.regularizer_args(),
        ],
    )
    parser.add_argument_group(
        'optimizer',
        description="optimizer's settings.",
        actions=[
            G_OPTIMIZER_ARG,
            *algorithm.optimizer_args(),
        ],
    )
    parser.set_defaults(creator_cls=algorithm)
    return parser


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

    @classmethod
    @abc.abstractmethod
    def model_args(cls) -> list[Action]:
        ...

    @classmethod
    @abc.abstractmethod
    def objective_args(cls) -> list[Action]:
        ...

    @classmethod
    @abc.abstractmethod
    def regularizer_args(cls) -> list[Action]:
        ...

    @classmethod
    @abc.abstractmethod
    def optimizer_args(cls) -> list[Action]:
        ...
