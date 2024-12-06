from __future__ import annotations

import functools
import pathlib
import time
import typing as t
import warnings

import numpy as np
from termcolor import colored

from core.evaluate import BLEUCalculator, FEDCalculator, SmoothingFunction, TextGenerator
from core.models.generators import Generator
from core.preprocess import MetaData, TextDataset
from core.train.callbacks import (
    CallbackList, ModelCheckpoint, ModelSaver, ProgbarLogger,
    TensorBoardXWritter, TextEvaluator, TrainProfiler,
)
from core.train.callbacks.channels import register_channel
from core.train.trainers import Trainer
from library.utils import SEPARATION_LINE, get_seqlens, logging_indent, random_sample


def create(
    args: _Args,
    trainer: Trainer,
    generator: Generator,
    data_collection: t.Mapping[str, TextDataset],
    metadata: MetaData,
    base_tag: str | None,
):
    base_tag = base_tag or f"{args.dataset}@{time.strftime('%Y%m%d-%H%M%S')}"
    creator = _CallbackCreator(
        generator=generator,
        data_collection=data_collection,
        metadata=metadata,
        tags=args.tags + [base_tag],
    )

    callback_list = CallbackList([
        creator.create_evaluator(
            bleu_n_gram=args.bleu,
            sample_size=args.batch_size,
            fed_sample_size=args.fed,
        ),
        *creator.create_loggers(
            updaters=trainer.updaters,
            tensorboard_logdir=args.tensorboard,
        ),
        *creator.create_savers(
            args,
            trainer=trainer,
            serving_root=args.serving_root,
            checkpoint_root=args.checkpoint_root,
            period=args.save_period,
        ),
        *creator.create_profiler(args.profile),
    ])
    callback_list.summary()
    return callback_list


class _Args(t.Protocol):
    dataset: str
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

    def __init__(self, generator, data_collection: t.Mapping[str, TextDataset], metadata: MetaData, tags: list[str]):
        self.generator = generator
        self.data_collection = data_collection
        self.metadata = metadata
        self.tag = pathlib.Path(*tags)

    def create_evaluator(self, bleu_n_gram: int | None, sample_size: int, fed_sample_size: int | None):
        return EvaluatorCreator(
            text_generator=self.text_generator,
            data_collection=self.data_collection,
            metadata=self.metadata,
        ).create(bleu_n_gram, sample_size, fed_sample_size)

    def create_loggers(self, updaters, tensorboard_logdir: pathlib.Path | None):
        yield ProgbarLogger(
            desc=str(self.tag),
            total=len(self.data_collection['train']),
            updaters=updaters,
        )
        if tensorboard_logdir:
            yield TensorBoardXWritter(
                updaters=updaters,
                logdir=tensorboard_logdir / self.tag,
                log_period=10,
            )

    def create_savers(
        self,
        args,
        trainer,
        serving_root: pathlib.Path | None,
        checkpoint_root: pathlib.Path | None,
        period: int,
    ):
        if serving_root:
            serving_dir = serving_root / self.tag
            serving_dir.mkdir(exist_ok=True)
            self.metadata.tokenizer.save(serving_dir / 'tokenizer.json')
            yield ModelSaver(
                module=self.text_generator,
                directory=serving_dir,
                period=period,
            )

        if checkpoint_root:
            yield ModelCheckpoint(
                args,
                trainer=trainer,
                directory=checkpoint_root / self.tag,
                period=period,
            )
        else:
            warnings.warn("`checkpoint_root` is not given. Training can't be restored!")

    def create_profiler(self, export_path: pathlib.Path | None):
        if export_path:
            yield TrainProfiler(
                warm_up=100,
                duration=200,
                export_filepath=export_path,
                stop_training_when_finish=True,
            )

    @functools.cached_property
    def text_generator(self):
        return TextGenerator(self.generator, tokenizer=self.metadata.tokenizer)


class EvaluatorCreator:

    def __init__(self, text_generator, data_collection: t.Mapping[str, TextDataset], metadata: MetaData):
        self.text_generator = text_generator
        self.data_collection = data_collection
        self.metadata = metadata

    def create(self, bleu_n_gram: int | None, sample_size: int, fed_sample_size: int | None):
        evaluator = TextEvaluator(self.text_generator)
        self._attach_basic(sample_size, evaluator)
        if bleu_n_gram is not None:
            self._attach_bleu(bleu_n_gram, sample_size, evaluator)
        if fed_sample_size is not None:
            self._attach_fed(fed_sample_size, evaluator)
        return evaluator

    def _attach_basic(self, sample_size, evaluator):

        def mean_length(word_ids):
            return {'mean_length': np.mean(get_seqlens(word_ids, self.metadata.eos_idx))}

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
            eos_idx=self.metadata.eos_idx,
            smoothing=SmoothingFunction.fuzz_smoothing,
        )
        for tag, dataset in self.data_collection.items():
            with logging_indent(f"Building '{tag}' data BLEU table..."):
                calculator = BLEUCalculator(
                    dataset.ids,
                    cache_dir=self.metadata.cache_dir / f"{tag}_BLEU",
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
