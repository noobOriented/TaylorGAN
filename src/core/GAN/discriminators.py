import typing as t

import torch

from core.models.interfaces import ModuleInterface
from core.models.sequence_modeling import TokenSequence
from core.objectives.collections import LossCollection


class Discriminator(torch.nn.Module, ModuleInterface):

    scope = 'Discriminator'

    def __init__(self, network: torch.nn.Module, embedder: torch.nn.Embedding):
        super().__init__()
        self.network = network
        self.embedder = embedder
        self.binary_output_layer = torch.nn.Linear(
            in_features=network(
                torch.zeros([1, 20, embedder.embedding_dim]),
            ).shape[-1],
            out_features=1,
        )

    def score_samples(self, samples: TokenSequence) -> torch.Tensor:
        word_vecs = self.get_embedding(samples.ids)
        return self.score_word_vector(word_vecs, samples.mask)

    def get_embedding(self, word_ids) -> torch.Tensor:
        return self.embedder(word_ids)

    def score_word_vector(self, word_vecs, mask=None) -> torch.Tensor:
        features = self.network(word_vecs, mask=mask)
        return self.binary_output_layer(features)

    @property
    def networks(self) -> list[torch.nn.Module]:
        return [self.embedder, self.network, self.binary_output_layer]

    @property
    def embedding_matrix(self):
        return self.embedder.weight


class DiscriminatorLoss(t.Protocol):

    def __call__(self, discriminator: Discriminator, real_samples, fake_samples) -> LossCollection:
        ...


class WordVectorRegularizer(DiscriminatorLoss):

    def __init__(self, max_norm: float = 0.):
        self.max_norm = max_norm

    def __call__(self, discriminator, real_samples, fake_samples):
        real_vecs = discriminator.get_embedding(real_samples.ids)
        fake_vecs = discriminator.get_embedding(fake_samples.ids)
        real_L2_loss = torch.square(real_vecs).sum(dim=-1)  # shape (N, T)
        fake_L2_loss = torch.square(fake_vecs).sum(dim=-1)  # shape (N, T)
        if self.max_norm:
            real_L2_loss = torch.maximum(real_L2_loss - self.max_norm ** 2, 0.)
            fake_L2_loss = torch.maximum(fake_L2_loss - self.max_norm ** 2, 0.)
        
        loss = (real_L2_loss + fake_L2_loss).mean() / 2
        return LossCollection(loss, word_vec=loss)


class GradientPenaltyRegularizer(DiscriminatorLoss):

    def __init__(self, center: float = 1.):
        self.center = center

    def __call__(self, discriminator, real_samples, fake_samples):
        real_vecs = discriminator.get_embedding(real_samples.ids)
        fake_vecs = discriminator.get_embedding(fake_samples.ids)

        eps = torch.rand(real_vecs.shape[0], 1, 1)
        inter_word_vecs = real_vecs * eps + fake_vecs * (1 - eps)
        score = discriminator.score_word_vector(inter_word_vecs)

        d_word_vecs, = torch.autograd.grad(
            score, inter_word_vecs,
            grad_outputs=torch.ones_like(score),
        )  # (N, T, E)
        grad_norm = torch.linalg.norm(d_word_vecs, dim=[1, 2])  # (N, )
        loss = torch.square(grad_norm - self.center).mean()
        return LossCollection(loss, grad_penalty=loss)
