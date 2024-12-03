import functools
import time
import warnings
import typing as t
from pathlib import Path

from core.evaluate import TextGenerator
from core.models.generators import Generator
from core.preprocess import MetaData
from core.preprocess.record_objects import TextDataset
from core.train.callbacks import (
    CallbackList, ModelCheckpoint, ModelSaver, ProgbarLogger, TensorBoardXWritter, TrainProfiler,
)
from core.train.trainers import Trainer

from .evaluator_creator import EvaluatorCreator


def create(
    args,
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
        *creator.create_profiler(export_path=args.profile),
    ])
    callback_list.summary()
    return callback_list


class _CallbackCreator:

    def __init__(self, generator, data_collection: t.Mapping[str, TextDataset], metadata: MetaData, tags: list[str]):
        self.generator = generator
        self.data_collection = data_collection
        self.metadata = metadata
        self.tag = Path(*tags)

    def create_evaluator(self, bleu_n_gram: int, sample_size: int, fed_sample_size: int):
        return EvaluatorCreator(
            text_generator=self.text_generator,
            data_collection=self.data_collection,
            metadata=self.metadata,
        ).create(bleu_n_gram, sample_size, fed_sample_size)

    def create_loggers(self, updaters, tensorboard_logdir: Path):
        yield ProgbarLogger(
            desc=self.tag,
            total=len(self.data_collection['train']),
            updaters=updaters,
        )
        if tensorboard_logdir:
            yield TensorBoardXWritter(
                updaters=updaters,
                logdir=tensorboard_logdir / self.tag,
                log_period=10,
            )

    def create_savers(self, args, trainer, serving_root: Path, checkpoint_root: Path, period: int):
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

    def create_profiler(self, export_path: Path):
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
