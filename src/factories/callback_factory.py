from __future__ import annotations

import functools
import pathlib
import time
import typing as t
import warnings

import numpy as np
import termcolor

from core.evaluate import BLEUCalculator, FEDCalculator, SmoothingFunction, TextGenerator
from core.models.generators import Generator
from core.preprocess import PreprocessResult
from core.train.callbacks import (
    Callback, CallbackList, ModelCheckpoint, ModelSaver,
    ProgbarLogger, TensorBoardXWritter, TextEvaluator, TrainProfiler,
)
from core.train.pubsub import register_channel
from core.train.trainers import Trainer
from library.utils import SEPARATION_LINE, get_seqlens, logging_indent, random_sample


def create(
    args: _Args,
    trainer: Trainer,
    generator: Generator,
    data: PreprocessResult,
    base_tag: str | None = None,
) -> Callback:
    creator = _CallbackCreator(
        args,
        data=data,
        generator=generator,
        trainer=trainer,
        base_tag=base_tag,
    )
    cbk = CallbackList(list(creator.create_callbacks()))
    cbk.summary()
    return cbk


class _Args(t.Protocol):
    batch_size: int

    # Evaluate
    bleu: int | None
    fed: int | None

    # Save
    checkpoint_root: pathlib.Path | None
    serving_root: pathlib.Path | None
    save_period: int

    # Logging
    tensorboard: pathlib.Path | None
    tags: list[str]

    # Dev
    profile: pathlib.Path | None



class _CallbackCreator:

    def __init__(
        self,
        args: _Args,
        generator: Generator,
        trainer: Trainer,
        data: PreprocessResult,
        base_tag: str | None = None,
    ):
        self.args = args
        self.data = data
        self.generator = generator
        self.trainer = trainer

        base_tag = base_tag or f"{data.cache_key}@{time.strftime('%Y%m%d-%H%M%S')}"
        self.tag = pathlib.Path(*self.args.tags, base_tag)

    def create_callbacks(self) -> t.Iterator[Callback]:
        yield self.create_evaluator()
        yield ProgbarLogger(
            desc=str(self.tag),
            total=len(self.data.dataset['train']),
            updaters=self.trainer.updaters,
        )
        if self.args.tensorboard:
            yield TensorBoardXWritter(
                updaters=self.trainer.updaters,
                logdir=self.args.tensorboard / self.tag,
                log_period=10,
            )

        if self.args.serving_root:
            serving_dir = self.args.serving_root / self.tag
            serving_dir.mkdir(parents=True, exist_ok=True)
            with open(serving_dir / 'tokenizer.json', 'w') as f:
                f.write(self.data.tokenizer.model_dump_json())
            yield ModelSaver(
                module=self.text_generator,
                directory=serving_dir,
                period=self.args.save_period,
            )

        if self.args.checkpoint_root:
            yield ModelCheckpoint(
                self.args,
                trainer=self.trainer,
                directory=self.args.checkpoint_root / self.tag,
                period=self.args.save_period,
            )
        else:
            warnings.warn("`checkpoint_root` is not given. Training can't be restored!")

        if self.args.profile:
            yield TrainProfiler(
                warm_up=100,
                duration=200,
                export_filepath=self.args.profile,
                stop_training_when_finish=True,
            )

    def create_evaluator(self):
        evaluator = TextEvaluator(self.text_generator)
        self._attach_basic(self.args.batch_size, evaluator)
        if self.args.bleu:
            self._attach_bleu(self.args.bleu, self.args.batch_size, evaluator)
        if self.args.fed:
            self._attach_fed(self.args.fed, evaluator)
        return evaluator

    @functools.cached_property
    def text_generator(self):
        return TextGenerator(self.generator, tokenizer=self.data.tokenizer)

    def _attach_basic(self, sample_size, evaluator: TextEvaluator):

        def mean_length(word_ids):
            return {'mean_length': np.mean(get_seqlens(word_ids, self.data.special_tokens.EOS.idx))}

        def log_texts(texts: list[str]):
            print(SEPARATION_LINE)
            print()
            print(termcolor.colored("Real Sentences (Random Sampled):", 'blue'))
            print_samples(random_sample(self.data.dataset['train'].texts, len(texts)))
            print()
            print(termcolor.colored("Fake Sentences (Random Sampled):", 'red'))
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

    def _attach_bleu(self, max_gram: int, sample_size: int, evaluator: TextEvaluator):
        for tag, dataset in self.data.dataset.items():
            with logging_indent(f"Building '{tag}' data BLEU table..."):
                calculator = BLEUCalculator(
                    dataset.ids,
                    cache_dir=pathlib.Path(self.data.cache_key, f'{tag}_BLEU'),
                    verbose=True,
                    max_gram=max_gram,
                    eos_idx=self.data.special_tokens.EOS.idx,
                    smoothing=SmoothingFunction.fuzz_smoothing,
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
            return BLEUCalculator.selfbleu(
                word_ids,
                max_gram=max_gram,
                eos_idx=self.data.special_tokens.EOS.idx,
                smoothing=SmoothingFunction.fuzz_smoothing,
            )

        evaluator.on_epoch_end.evaluate_ids(
            selfbleu,
            sample_size=min(10000, 2 * len(self.data.dataset['train'])),
            channel=register_channel('samples'),
        )

    def _attach_fed(self, sample_size: int, evaluator: TextEvaluator):
        for tag, dataset in self.data.dataset.items():
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
