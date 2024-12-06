from __future__ import annotations

import functools
import typing as t

import pydantic
import torch
from torch.nn import Embedding, Linear

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
from core.preprocess import MetaData
from core.train import DiscriminatorUpdater, GANTrainer, GeneratorUpdater, NonParametrizedTrainer
from core.train.optimizer import OptimizerWrapper
from library.torch_zoo.nn import LambdaModule, activations
from library.torch_zoo.nn.masking import (
    MaskAvgPool1d, MaskConv1d, MaskGlobalAvgPool1d, MaskSequential,
)
from library.torch_zoo.nn.resnet import ResBlock
from library.utils import ArgumentBinder, LookUpCall, NamedObject


class MLEObjectiveConfigs(pydantic.BaseModel):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    g_regularizers: list[str] = []

    def get_trainer(self, metadata: MetaData, generator: Generator):
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

    def get_trainer(self, metadata: MetaData, generator: Generator):
        discriminator = self._create_discriminator(metadata)
        objective = GANObjective(
            discriminator=discriminator,
            generator_loss=self._loss_tuple.generator_loss,
            estimator=_ESTIMATORS(self.estimator),
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

    def _create_discriminator(self, metadata: MetaData) -> Discriminator:
        print(f"Create discriminator: {self.discriminator}")
        network_func, arginfo = _D_MODELS(self.discriminator, return_info=True)
        embedder = Embedding.from_pretrained(
            torch.from_numpy(metadata.load_pretrained_embeddings()),
            freeze=self.d_fix_embeddings,
        )
        return NamedObject(
            Discriminator(
                network=network_func(embedder.embedding_dim),
                embedder=embedder,
            ),
            name=arginfo.func_name,
        )

    @functools.cached_property
    def _loss_tuple(self) -> GANLossTuple:
        return {
            'alt': GANLossTuple(lambda fake_score: BCE(fake_score, labels=1.)),  # RKL - 2JS
            'JS': GANLossTuple(lambda fake_score: -BCE(fake_score, labels=0.)),  # 2JS
            'KL': GANLossTuple(lambda fake_score: -torch.exp(fake_score)),  # -sig / (1 - sig)
            'RKL': GANLossTuple(lambda fake_score: -fake_score),  # log((1 - sig) / sig)
        }[self.loss]


def cnn(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        MaskConv1d(input_size, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskConv1d(512, 512, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskAvgPool1d(kernel_size=2),
        MaskConv1d(512, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskConv1d(1024, 1024, kernel_size=3, padding=1),
        ActivationLayer(),
        MaskGlobalAvgPool1d(),
        Linear(1024, 1024),
        ActivationLayer(),
    )


def resnet(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        Linear(input_size, 512),
        ActivationLayer(),
        LambdaModule(lambda x: torch.transpose(x, 1, 2)),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        ResBlock(512, kernel_size=3),
        ActivationLayer(),
        MaskGlobalAvgPool1d(),
        Linear(512, 512),
        ActivationLayer(),
    )


_G_REGS = LookUpCall({
    'spectral': LossScaler.as_constructor(SpectralRegularizer),
    'embedding': LossScaler.as_constructor(EmbeddingRegularizer),
    'entropy': LossScaler.as_constructor(EntropyRegularizer),
})
_OPTIMIZERS = LookUpCall({
    key: ArgumentBinder(OptimizerWrapper.as_constructor(optim_cls), preserved=['params'])
    for key, optim_cls in [
        ('sgd', torch.optim.SGD),
        ('rmsprop', torch.optim.RMSprop),
        ('adam', torch.optim.Adam),
    ]
})

_D_MODELS = LookUpCall({
    key: ArgumentBinder(func, preserved=['input_size'])
    for key, func in [
        ('cnn', cnn),
        ('resnet', resnet),
        ('test', lambda input_size: MaskGlobalAvgPool1d(dim=1)),
    ]
})
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
