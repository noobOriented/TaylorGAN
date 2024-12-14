import typing as t

import torch

from core.models import AutoRegressiveGenerator, Generator, TokenSequence
from library.torch_zoo.functions import masked_reduce


def mean_negative_log_likelihood(generator: AutoRegressiveGenerator, real_samples: TokenSequence):
    return generator.seq_neg_logprobs(real_samples.ids).mean()


class GeneratorLoss(t.Protocol):

    def __call__(self, generator: Generator, real_samples: TokenSequence) -> torch.Tensor:
        ...


class EntropyRegularizer(GeneratorLoss):

    def __call__(self, generator: AutoRegressiveGenerator, real_samples: TokenSequence):
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        # NOTE it's biased
        logp = torch.nn.functional.log_softmax(fake_samples.logits, dim=-1)  # (N, T, V)
        neg_entropy = (logp.detach() * fake_samples.probs).sum(dim=-1)  # (N, T)
        # TODO observable: entropy=fake_samples.seq_neg_logprobs.mean()
        return masked_reduce(neg_entropy, mask=fake_samples.mask)  # scalar
