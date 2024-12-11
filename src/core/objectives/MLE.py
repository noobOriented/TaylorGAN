from core.models import Generator
from core.models.sequence_modeling import TokenSequence


class MLEObjective:

    def __call__(self, generator: Generator, real_samples: TokenSequence):
        return generator.seq_neg_logprobs(real_samples.ids).mean()
