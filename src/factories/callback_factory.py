from __future__ import annotations

import functools
import pathlib
import time
import typing as t
import warnings

from core.evaluate import TextGenerator
from core.models.generators import Generator
from core.preprocess import PreprocessResult
from core.train.callbacks import (
    BLEUEvaluator, Callback, CallbackList, FEDEvaluator, ModelCheckpoint,
    ModelSaver, ProgbarLogger, TensorBoardXWritter, TextEvaluator, TrainProfiler,
)
from core.train.fit_loop import DataLoader
from core.train.trainers import Trainer


def create(
    args: _Args,
    trainer: Trainer,
    generator: Generator,
    data: PreprocessResult,
    data_loader: DataLoader,
    base_tag: str | None = None,
):
    creator = _CallbackCreator(
        args,
        data=data,
        generator=generator,
        trainer=trainer,
        base_tag=base_tag,
        data_loader=data_loader,
    )
    for cbk in creator.create_callbacks():
        data_loader._callback.on_train_begin.attach(cbk.on_train_begin)
        data_loader._callback.on_epoch_begin.attach(cbk.on_epoch_begin)
        data_loader._callback.on_batch_begin.attach(cbk.on_batch_begin)
        data_loader._callback.on_batch_end.attach(cbk.on_batch_end)
        data_loader._callback.on_epoch_end.attach(cbk.on_epoch_end)
        data_loader._callback.on_train_end.attach(cbk.on_train_end)


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
        data_loader: DataLoader,
        base_tag: str | None = None,
    ):
        self.args = args
        self.data = data
        self.generator = generator
        self.trainer = trainer

        base_tag = base_tag or f"{data.cache_key}@{time.strftime('%Y%m%d-%H%M%S')}"
        self.tag = pathlib.Path(*self.args.tags, base_tag)

    def create_callbacks(self) -> t.Iterator[Callback]:
        yield from self.create_evaluators()
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

    def create_evaluators(self):
        yield TextEvaluator(
            self.text_generator,
            self.data.special_tokens.EOS.idx,
            self.data.dataset['train'].texts,
        )
        if self.args.bleu:
            yield BLEUEvaluator(
                self.data,
                self.text_generator,
                self.args.bleu,
                self.args.batch_size,
            )

        if self.args.fed:
            yield FEDEvaluator(self.text_generator, self.data, self.args.fed)

    @functools.cached_property
    def text_generator(self):
        return TextGenerator(self.generator, tokenizer=self.data.tokenizer)


def print_samples(texts: t.Sequence[str]):
    for i, line in enumerate(texts, 1):
        print(f"{i}.")
        print(line)
