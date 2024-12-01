import typing as t

import numpy as np
from termcolor import colored

from core.evaluate import BLEUCalculator, FEDCalculator, SmoothingFunction
from core.preprocess.record_objects import TextDataset
from core.train.callbacks import TextEvaluator
from core.train.callbacks.channels import register_channel
from library.utils import SEPARATION_LINE, get_seqlens, logging_indent, random_sample


class EvaluatorCreator:

    def __init__(self, text_generator, data_collection: t.Mapping[str, TextDataset], meta_data):
        self.text_generator = text_generator
        self.data_collection = data_collection
        self.meta_data = meta_data

    def create(self, bleu_n_gram, sample_size, fed_sample_size):
        evaluator = TextEvaluator(self.text_generator)
        self._attach_basic(sample_size, evaluator)
        if bleu_n_gram is not None:
            self._attach_bleu(bleu_n_gram, sample_size, evaluator)
        if fed_sample_size is not None:
            self._attach_fed(fed_sample_size, evaluator)
        return evaluator

    def _attach_basic(self, sample_size, evaluator):

        def mean_length(word_ids):
            return {'mean_length': np.mean(get_seqlens(word_ids, self.meta_data.eos_idx))}

        def log_texts(texts: list[str]):
            print(SEPARATION_LINE)
            print()
            print(colored("Real Sentences (Random Sampled):", 'blue'))
            print_samples(random_sample(self.data_collection['train'].texts, len(texts)))
            print()
            print(colored("Fake Sentences (Random Sampled):", 'red'))
            print_samples(texts)
            print()

        evaluator.on_batch_end.evaluate_ids(
            mean_length,
            sample_size=sample_size,
            channel=register_channel('samples'),
            period=10,
        )
        evaluator.on_batch_end.evaluate_texts(
            log_texts,
            sample_size=3,
            period=100,
        )

    def _attach_bleu(self, max_gram, sample_size, evaluator):
        shared_kwargs = dict(
            max_gram=max_gram,
            eos_idx=self.meta_data.eos_idx,
            smoothing=SmoothingFunction.fuzz_smoothing,
        )
        for tag, dataset in self.data_collection.items():
            with logging_indent(f"Building '{tag}' data BLEU table..."):
                calculator = BLEUCalculator(
                    dataset.ids,
                    cache_dir=self.meta_data.cache_dir / f"{tag}_BLEU",
                    verbose=True,
                    **shared_kwargs,
                )
            evaluator.on_batch_end.evaluate_ids(
                calculator.mean_bleu,
                sample_size=sample_size,
                channel=register_channel(tag),
                period=10,
            )

        def selfbleu(word_ids) -> t.Callable:
            print("Evaluating generated data SelfBLEU...")
            print()
            return BLEUCalculator.selfbleu(word_ids, **shared_kwargs)

        evaluator.on_epoch_end.evaluate_ids(
            selfbleu,
            sample_size=min(10000, 2 * len(self.data_collection['train'])),
            channel=register_channel('samples'),
        )

    def _attach_fed(self, sample_size, evaluator):
        for tag, dataset in self.data_collection.items():
            print(f"Building '{tag}' data FED sentence encoder...")
            calculator = FEDCalculator(
                hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                references=random_sample(dataset.texts, size=sample_size),
            )

            def fed(texts):
                print("Evaluating FED Score ...")
                print()
                return {"FED": calculator.calculate_fed_score(candidates=texts)}

            evaluator.on_epoch_end.evaluate_texts(
                fed,
                sample_size=sample_size,
                channel=register_channel(tag),
            )


def print_samples(texts: t.Sequence[str]):
    for i, line in enumerate(texts, 1):
        print(f"{i}.")
        print(line)
