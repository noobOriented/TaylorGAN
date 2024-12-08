from __future__ import annotations

import os
import pathlib
import sys
import time
import typing as t
import warnings

import numpy as np
import termcolor
import torch

from core.evaluate import TextGenerator
from core.models.generators import Generator
from core.preprocess import PreprocessResult
from core.train import Callback, ModelCheckpointSaver, Trainer
from core.train.loggers import ProgbarLogger
from core.train.pubsub import METRIC_CHANNELS, register_channel
from library.utils import SEPARATION_LINE, get_seqlens, logging_indent, random_sample


def create(
    args: _Args,
    trainer: Trainer,
    generator: Generator,
    data: PreprocessResult,
    checkpoint: str | os.PathLike[str] | None = None,
    base_tag: str | None = None,
):
    creator = _CallbackCreator(
        args,
        data=data,
        generator=generator,
        trainer=trainer,
        checkpoint=checkpoint,
        base_tag=base_tag,
    )
    creator.attach_events()
    return creator.callback


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
        checkpoint: str | os.PathLike[str] | None,
        base_tag: str | None = None,
    ):
        self.args = args
        self.data = data
        self.generator = generator
        self.trainer = trainer
        self.checkpoint = checkpoint
        self.callback = Callback()

        base_tag = base_tag or f"{data.cache_key}@{time.strftime('%Y%m%d-%H%M%S')}"
        self.tag = pathlib.Path(*self.args.tags, base_tag)
        self.text_generator = TextGenerator(self.generator, tokenizer=self.data.tokenizer)

    def attach_events(self):
        self._attach_evaluators()
        self._attach_loggers()
        self._attach_savers()

        if export_path := self.args.profile:
            import cProfile
            import pstats
            profile = cProfile.Profile(subcalls=False)
            warm_up = 100
            duration = 200

            @self.callback.on_batch_begin.attach
            def enable_profile(batch: int):
                if batch == 100:
                    print(f"Updates {warm_up} times.")
                    print("Complete warm-up, start profiling.")
                    profile.enable()

            @self.callback.on_batch_end.attach
            def disable_profile(batch: int, _):
                if batch != warm_up + duration:
                    return

                profile.disable()
                print(f"Updates {warm_up} + {duration} times.")
                print(f"Complete profiling, export stats to {export_path}")
                with open(export_path, 'w') as f_out:
                    stats = pstats.Stats(profile, stream=f_out)
                    stats.strip_dirs().sort_stats('cumtime').print_stats()

                print("Exit by TrainProfiler.")
                sys.exit(0)

    def _attach_evaluators(self):

        @self.callback.on_batch_end.attach
        def calculate_mean_length(batch: int, _, hook=register_channel('samples')):
            if batch % 10 == 0:
                ids = self.text_generator.generate_ids(10)
                mean_length = np.mean(get_seqlens(ids, self.data.special_tokens.EOS.idx))
                hook(batch, {'mean_length': float(mean_length)})

        @self.callback.on_batch_end.attach
        def log_texts(batch: int, _):
            if batch % 100 == 0:
                sentences = self.text_generator.generate_texts(3)
                print(SEPARATION_LINE)
                print()
                print(termcolor.colored("Real Sentences (Random Sampled):", 'blue'))
                _print_samples(random_sample(self.data.dataset['train'].texts, len(sentences)))
                print()
                print(termcolor.colored("Fake Sentences (Random Sampled):", 'red'))
                _print_samples(sentences)
                print()

        if ngram := self.args.bleu:
            from core.evaluate import BLEUCalculator, SmoothingFunction

            for tag, dataset in self.data.dataset.items():
                with logging_indent(f"Building '{tag}' data BLEU table..."):
                    calculator = BLEUCalculator(
                        dataset.ids,
                        cache_dir=pathlib.Path(self.data.cache_key, f'{tag}_BLEU'),
                        verbose=True,
                        max_gram=ngram,
                        eos_idx=self.data.special_tokens.EOS.idx,
                        smoothing=SmoothingFunction.fuzz_smoothing,
                    )

                @self.callback.on_batch_end.attach
                def bleu(batch: int, _, /, calculator=calculator, hook=register_channel(tag)):
                    if batch % 10 == 0:
                        ids = self.text_generator.generate_ids(self.args.batch_size)
                        result = calculator.mean_bleu(ids)
                        hook(batch, result)

            @self.callback.on_epoch_end.attach
            def self_bleu(epoch: int, hook=register_channel('samples')):
                ids = self.text_generator.generate_ids(len(self.data.dataset['train']))

                print("Evaluating generated data SelfBLEU...")
                print()
                selfbleu = BLEUCalculator.selfbleu(
                    ids,
                    max_gram=ngram,
                    eos_idx=self.data.special_tokens.EOS.idx,
                    smoothing=SmoothingFunction.fuzz_smoothing,
                )
                hook(epoch, selfbleu)

        if fed_sample_size := self.args.fed:
            from core.evaluate import FEDCalculator

            for tag, dataset in self.data.dataset.items():
                print(f"Building '{tag}' data FED sentence encoder...")
                calculator = FEDCalculator(
                    hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                    references=random_sample(dataset.texts, size=fed_sample_size),
                )

                @self.callback.on_epoch_end.attach
                def fed(epoch: int, calculator=calculator, hook=register_channel(tag)):
                    texts = self.generator.generate_texts(fed_sample_size)
                    d = {'FED': calculator.calculate_fed_score(texts)}
                    hook(epoch, d)

    def _attach_loggers(self):
        bar = ProgbarLogger(
            desc=str(self.tag),
            total=len(self.data.dataset['train']),
            updaters=self.trainer.updaters,
        )
        self.callback.on_train_begin.attach(bar.on_train_begin)
        self.callback.on_epoch_begin.attach(bar.on_epoch_begin)
        self.callback.on_batch_end.attach(bar.on_batch_end)
        self.callback.on_epoch_end.attach(bar.on_epoch_end)
        self.callback.on_train_end.attach(bar.on_train_end)

        if self.args.tensorboard:
            from tensorboardX import SummaryWriter

            logdir= self.args.tensorboard / self.tag
            writer = SummaryWriter(logdir=str(logdir))

            @self.callback.on_train_begin.attach
            def on_train_begin(is_restored: bool):
                logdir.mkdir(exist_ok=True)
                writer.add_text(
                    'restore_args' if self.checkpoint else 'args',
                    ' '.join(sys.argv[1:]),
                    0,
                )

            for updater in self.trainer.updaters:
                @updater.hook.attach
                def update_losses(step: int, losses: t.Mapping):
                    if step % 10 == 0:
                        for key, val in losses.items():
                            writer.add_scalar(
                                tag=f'losses/{updater.module.scope}/{key}',
                                scalar_value=val,
                                global_step=step,  # TODO enerator step?
                            )

            for name, channel in METRIC_CHANNELS.items():
                @channel.attach
                def update_metrics(step: int, vals: t.Mapping):
                    for key, val in vals.items():
                        writer.add_scalar(
                            tag=f'{name}/{key}',
                            scalar_value=val,
                            global_step=step,  # TODO batch ? epoch?
                        )

    def _attach_savers(self):
        if self.args.serving_root:
            serving_dir = self.args.serving_root / self.tag
            serving_dir.mkdir(parents=True, exist_ok=True)
            (serving_dir / 'tokenizer.json').write_text(self.data.tokenizer.model_dump_json())

            @self.callback.on_epoch_end.attach
            def save_torch(epoch):
                if epoch % self.args.save_period == 0:
                    path = serving_dir / f"model_epo{epoch}.pth"
                    print(f"{epoch} epochs done. Save model to {path}.")
                    traced = self.text_generator.export_traced()
                    torch.jit.save(traced, str(path))

        if self.args.checkpoint_root:
            saver = ModelCheckpointSaver(
                trainer=self.trainer,
                directory=self.args.checkpoint_root / self.tag,
            )

            @self.callback.on_train_begin.attach
            def save_args():
                saver.directory.mkdir(exist_ok=True)
                if not self.checkpoint:
                    with open(saver.directory / 'args', 'w') as f:
                        f.write(self.args.model_dump_json())

            @self.callback.on_epoch_end.attach
            def save_model(epoch: int):
                if epoch % self.args.save_period == 0:
                    print(f"{epoch} epochs done.")
                    saver.save(epoch)

        else:
            warnings.warn("`checkpoint_root` is not given. Training can't be restored!")


def _print_samples(texts: t.Sequence[str], /):
    for i, line in enumerate(texts, 1):
        print(f"{i}.", line)
