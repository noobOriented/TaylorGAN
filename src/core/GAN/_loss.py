from __future__ import annotations

import abc
import dataclasses
import typing as t

import torch

from core.models import TokenSequence
from core.train import GeneratorLoss
from library.torch_zoo.functions import gaussian, masked_reduce, pairwise_euclidean
from library.utils import format_object

from ._discriminator import Discriminator


@dataclasses.dataclass
class GANObjective(GeneratorLoss):
    discriminator: Discriminator
    generator_loss: t.Callable[[torch.Tensor], torch.Tensor]
    estimator: GANEstimator

    def __call__(self, generator, real_samples):
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        return self.estimator.compute_loss(
            fake_samples,
            discriminator=self.discriminator,
            generator_loss=self.generator_loss,
        )

    def __str__(self):
        return format_object(self, estimator=self.estimator)


class GANEstimator(abc.ABC):

    @abc.abstractmethod
    def compute_loss(
        self,
        fake_samples: TokenSequence,
        discriminator: Discriminator,
        generator_loss: t.Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        ...

    def __str__(self):
        return format_object(self)


class GANLossTuple:

    def __init__(self, generator_loss: t.Callable[[torch.Tensor], torch.Tensor]):
        self.generator_loss = generator_loss

    def discriminator_loss(self, discriminator: Discriminator, real_samples: TokenSequence, fake_samples: TokenSequence):
        real_score = discriminator.score_samples(real_samples)
        fake_score = discriminator.score_samples(fake_samples)
        loss_real = BCE(real_score, labels=1.)
        loss_fake = BCE(fake_score, labels=0.)
        return (loss_real + loss_fake).mean()


class SoftmaxEstimator(GANEstimator):

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        return _compute_loss_of_probability(
            discriminator,
            generator_loss,
            probs=fake_samples.probs,
            mask=fake_samples.mask,
        )


class GumbelSoftmaxEstimator(GANEstimator):

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        return _compute_loss_of_probability(
            discriminator,
            generator_loss,
            probs=torch.nn.functional.softmax(fake_samples.logits + fake_samples.gumbel_vars),
            mask=fake_samples.mask,
        )


class ReinforceEstimator(GANEstimator):

    def __init__(self, baseline_decay: float = 0.9):
        self.baseline_decay = baseline_decay
        self.baseline = None

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        score = discriminator.score_samples(fake_samples)
        adv_loss = generator_loss(score)
        reward = adv_loss.squeeze(axis=1)  # shape (N, )

        advantage = self.compute_advantage(reward)  # shape (N, )
        # TODO observable adv=adv_loss.mean()
        return (advantage.detach() * fake_samples.seq_neg_logprobs).mean()

    def compute_advantage(self, reward):
        if self.baseline is None:
            self.baseline = reward.mean()

        advantage = reward - self.baseline
        self.baseline = (
            self.baseline * self.baseline_decay + reward.mean() * (1 - self.baseline_decay)
        )
        return advantage


class TaylorEstimator(GANEstimator):

    def __init__(self, baseline_decay: float = 0.9, bandwidth: float = 0.5):
        self.baseline_decay = baseline_decay
        self.bandwidth = bandwidth
        self.baseline = None

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        fake_embeddings = discriminator.get_embedding(word_ids=fake_samples.ids)
        score = discriminator.score_word_vector(fake_embeddings, mask=fake_samples.mask)
        adv_loss = generator_loss(score)
        reward = -adv_loss

        first_order_reward = self.taylor_first_order(
            y=reward,
            x0=fake_embeddings,
            xs=discriminator.embedding_weight,
        ).view_as(fake_samples.logits)
        zeroth_order_advantage = self.compute_advantage(reward)
        advantage = zeroth_order_advantage.unsqueeze(dim=2) + first_order_reward

        square_dist = pairwise_euclidean(discriminator.embedding_weight)
        kernel = gaussian(square_dist / (self.bandwidth ** 2))  # (V, V)
        batch_kernel = torch.nn.functional.embedding(fake_samples.ids, kernel)  # shape (N, T, V)
        likelihood = torch.tensordot(fake_samples.probs, kernel, dims=1)

        normalized_advantage = batch_kernel * advantage / (likelihood + 1e-8)
        full_loss = -normalized_advantage.detach() * fake_samples.probs
        # TODO observable adv=adv_loss.mean()
        return masked_reduce(full_loss, mask=fake_samples.mask)

    @staticmethod
    def taylor_first_order(y, x0, xs):
        """
        Args:
            y: any shape computed by x0
            x0: shape (*M, d)
            xs: shape (N, d)

        Returns:
            dy: shape (*M, N)
        """
        dydx0, = torch.autograd.grad(y, x0, grad_outputs=torch.ones_like(y))  # (*M, d)
        # dydx0 * (xs - x0) = dydx0 * xs - dydx0 * x0
        return (
            torch.tensordot(dydx0, xs, dims=[[-1], [-1]])  # (*M, N)
            - (dydx0 * x0).sum(dim=-1, keepdim=True)  # (*M, 1)
        )

    def compute_advantage(self, reward):
        if self.baseline is None:
            self.baseline = reward.mean()

        advantage = reward - self.baseline
        self.baseline = (
            self.baseline * self.baseline_decay + reward.mean() * (1 - self.baseline_decay)
        )
        return advantage


class StraightThroughEstimator(GANEstimator):

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        word_vecs = discriminator.get_embedding(word_ids=fake_samples.ids)
        score = discriminator.score_word_vector(word_vecs)
        adv_loss = generator_loss(score)

        d_word_vecs, = torch.autograd.grad(adv_loss, word_vecs)  # (N, T, E)
        # NOTE, can be derived by chain-rule
        d_onehot = torch.tensordot(d_word_vecs, discriminator.embedding_weight, dims=[[-1], [-1]])
        full_loss = d_onehot.detach() * fake_samples.probs  # (N, T, V)
        # TODO observable adv=adv_loss.mean()
        return masked_reduce(full_loss, mask=fake_samples.mask)


def _compute_loss_of_probability(
    discriminator: Discriminator, generator_loss, probs, mask):
    word_vecs = torch.tensordot(probs, discriminator.embedding_weight, dims=1)  # (N, T, E)
    score = discriminator.score_word_vector(word_vecs, mask)
    return generator_loss(score).mean()



def D_BCE(real_score, fake_score):
    loss_real = BCE(real_score, labels=1.)
    loss_fake = BCE(fake_score, labels=0.)
    return (loss_real + loss_fake).mean()


def BCE(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        score,
        target=torch.full_like(score, labels),
    )
