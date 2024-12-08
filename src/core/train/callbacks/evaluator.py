import pathlib
import typing as t

import numpy as np
import termcolor

from core.evaluate import BLEUCalculator, FEDCalculator, SmoothingFunction, TextGenerator
from core.preprocess import PreprocessResult
from library.utils import SEPARATION_LINE, get_seqlens, logging_indent, random_sample

from ..pubsub import register_channel
from .base import Callback


class TextEvaluator(Callback):

    def __init__(self, generator: TextGenerator, eos_idx: int, real_samples: t.Sequence[str]):
        self.generator = generator
        self.eos_idx = eos_idx
        self.real_samples = real_samples
        self.channel = register_channel('samples')

    def on_batch_end(self, batch: int, batch_data):
        if batch % 10 == 0:
            ids = self.generator.generate_ids(10)
            mean_length = np.mean(get_seqlens(ids, self.eos_idx))
            self.channel.notify(batch, {'mean_length': mean_length})

        if batch % 100 == 0:
            sentenses = self.generator.generate_texts(3)
            print(SEPARATION_LINE)
            print()
            print(termcolor.colored("Real Sentences (Random Sampled):", 'blue'))
            self._print_samples(random_sample(self.real_samples, len(sentenses)))
            print()
            print(termcolor.colored("Fake Sentences (Random Sampled):", 'red'))
            self._print_samples(sentenses)
            print()

    def _print_samples(self, texts: t.Sequence[str], /):
        for i, line in enumerate(texts, 1):
            print(f"{i}.")
            print(line)


class BLEUEvaluator(Callback):

    def __init__(
        self,
        data: PreprocessResult,
        generator: TextGenerator,
        max_gram: int,
        sample_size: int,
        period: int = 10,
    ):
        self.channels = {}
        self.calculators: dict[str, BLEUCalculator] = {}
        for tag, dataset in data.dataset.items():
            with logging_indent(f"Building '{tag}' data BLEU table..."):
                self.calculators[tag] = BLEUCalculator(
                    dataset.ids,
                    cache_dir=pathlib.Path(data.cache_key, f'{tag}_BLEU'),
                    verbose=True,
                    max_gram=max_gram,
                    eos_idx=data.special_tokens.EOS.idx,
                    smoothing=SmoothingFunction.fuzz_smoothing,
                )
                self.channels[tag] = register_channel(tag)

        self.channels['samples'] = register_channel('samples')
        self.max_gram = max_gram
        self.generator = generator
        self.sample_size = sample_size
        self.period = period
        self.selfbleu_sample_size = len(data.dataset['train'])
        self.eos_idx = data.special_tokens.EOS.idx

    def on_batch_end(self, batch: int, batch_data):
        if batch % self.period == 0:
            ids = self.generator.generate_ids(self.sample_size)
            for tag, c in self.calculators.items():
                result = c.mean_bleu(ids)
                self.channels[tag].notify(batch, result)

    def on_epoch_end(self, epoch: int):
        ids = self.generator.generate_ids(self.selfbleu_sample_size)

        print("Evaluating generated data SelfBLEU...")
        print()
        selfbleu = BLEUCalculator.selfbleu(
            ids,
            max_gram=self.max_gram,
            eos_idx=self.eos_idx,
            smoothing=SmoothingFunction.fuzz_smoothing,
        )
        self.channels['samples'].notify(epoch, selfbleu)


class FEDEvaluator(Callback):
    
    def __init__(self, generator: TextGenerator, data: PreprocessResult, sample_size: int) -> None:
        self.generator = generator
        self.sample_size = sample_size
        self.calculators: dict[str, FEDCalculator] = {}
        for tag, dataset in data.dataset.items():
            print(f"Building '{tag}' data FED sentence encoder...")
            self.calculators[tag] = FEDCalculator(
                hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                references=random_sample(dataset.texts, size=sample_size),
            )

    def on_epoch_end(self, epoch: int):
        texts = self.generator.generate_texts(self.sample_size)
        for tag, c in self.calculators.items():
            d = {'FED': c.calculate_fed_score(texts)}
            register_channel(tag).notify(epoch, d)
