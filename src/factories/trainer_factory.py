from __future__ import annotations

import functools
import typing as t

import pydantic
import torch

from core.GAN import (
    BCE, Discriminator, DiscriminatorUpdater, GANLossTuple, GANObjective, GANTrainer,
    GradientPenaltyRegularizer, GumbelSoftmaxEstimator, ReinforceEstimator,
    StraightThroughEstimator, TaylorEstimator, WordVectorRegularizer,
)
from core.GAN.discriminators import DiscriminatorLoss, EmbeddingRegularizer, SpectralRegularizer
from core.losses import EntropyRegularizer, GeneratorLoss, mean_negative_log_likelihood
from core.models import Generator
from core.preprocess import PreprocessResult
from core.train import GeneratorUpdater, NonParametrizedTrainer
from core.train.optimizer import add_custom_optimizer_args
from library.torch_zoo.nn import LambdaModule, activations
from library.torch_zoo.nn.masking import (
    MaskAvgPool1d, MaskConv1d, MaskGlobalAvgPool1d, MaskSequential,
)
from library.torch_zoo.nn.resnet import ResBlock
from library.utils import ArgumentBinder, LookUpCall, wraps_with_new_signature


class MLEObjectiveConfigs(pydantic.BaseModel):
    g_optimizer: str = 'adam(lr=1e-4,betas=(0.5, 0.999),clip_norm=10)'
    g_regularizers: list[str] = []

    def get_trainer(self, data: PreprocessResult, generator: Generator):
        losses: dict[str, tuple[GeneratorLoss, float]] = {'NLL': (mean_negative_log_likelihood, 1)}
        for s in self.g_regularizers:
            (reg, coeff), info = _G_REGS(s, return_info=True)
            losses[info.func_name] = (reg, coeff)

        generator_updater = GeneratorUpdater(
            generator,
            optimizer=_OPTIMIZERS(self.g_optimizer)(generator.parameters()),
            losses=losses,
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

    def get_trainer(self, data: PreprocessResult, generator: Generator):
        discriminator = self._create_discriminator(data)
        objective = GANObjective(
            discriminator=discriminator,
            generator_loss=self._loss_tuple.generator_loss,
            estimator=_ESTIMATORS(self.estimator),
        )
        g_losses: dict[str, tuple[GeneratorLoss, float]] = {
            self.loss: (objective, 1),
        }
        for s in self.g_regularizers:
            (reg, coeff), info = _G_REGS(s, return_info=True)
            g_losses[info.func_name] = (reg, coeff)

        d_losses: dict[str, tuple[DiscriminatorLoss, float]] = {
            'BCE': (self._loss_tuple.discriminator_loss, 1),
        }
        for s in self.d_regularizers:
            (reg, coeff), info = _D_REGS(s, return_info=True)
            d_losses[info.func_name] = (reg, coeff)

        generator_updater = GeneratorUpdater(
            generator,
            optimizer=_OPTIMIZERS(self.g_optimizer)(generator.parameters()),
            losses=g_losses,
        )
        return GANTrainer(
            generator_updater=generator_updater,
            discriminator_updater=DiscriminatorUpdater(
                discriminator,
                optimizer=_OPTIMIZERS(self.d_optimizer)(discriminator.parameters()),
                losses=d_losses,
            ),
            d_steps=self.d_steps,
        )

    def _create_discriminator(self, data: PreprocessResult) -> Discriminator:
        print(f"Create discriminator: {self.discriminator}")
        network_func = _D_MODELS(self.discriminator)
        embedder = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(data.embedding_matrix),
            freeze=self.d_fix_embeddings,
            padding_idx=data.special_tokens.PAD.idx,
        )
        return Discriminator(
            network=network_func(embedder.embedding_dim),
            embedder=embedder,
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
        torch.nn.Linear(1024, 1024),
        ActivationLayer(),
    )


def resnet(input_size, activation: activations.TYPE_HINT = 'relu'):
    ActivationLayer = activations.deserialize(activation)
    return MaskSequential(
        torch.nn.Linear(input_size, 512),
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
        torch.nn.Linear(512, 512),
        ActivationLayer(),
    )


def _concat_coeff[**P, T](
    regularizer_cls: t.Callable[P, T],
) -> t.Callable[t.Concatenate[float, P], tuple[T, float]]:

    @wraps_with_new_signature(regularizer_cls)
    def wrapper(coeff: float, *args: P.args, **kwargs: P.kwargs):
        return regularizer_cls(*args, **kwargs), coeff

    return wrapper


_G_REGS = LookUpCall({
    'entropy': _concat_coeff(EntropyRegularizer),
})
_OPTIMIZERS = LookUpCall({
    key: ArgumentBinder(add_custom_optimizer_args(optim_cls), preserved=['params'])
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
    'spectral': _concat_coeff(SpectralRegularizer),
    'embedding': _concat_coeff(EmbeddingRegularizer),
    'grad_penalty': _concat_coeff(GradientPenaltyRegularizer),
    'word_vec': _concat_coeff(WordVectorRegularizer),
})
