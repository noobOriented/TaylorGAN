from __future__ import annotations

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
from core.train.callbacks import Callback, CustomCallback, ModelCheckpoint, ProgbarLogger
from core.train.pubsub import CHANNELS, register_channel
from core.train.trainers import Trainer
from library.utils import SEPARATION_LINE, get_seqlens, logging_indent, random_sample


def create(
    args: _Args,
    trainer: Trainer,
    generator: Generator,
    data: PreprocessResult,
    base_tag: str | None = None,
):
    creator = _CallbackCreator(
        args,
        data=data,
        generator=generator,
        trainer=trainer,
        base_tag=base_tag,
    )
    for cbk in creator.create_callbacks():
        creator.callback.on_train_begin.attach(cbk.on_train_begin)
        creator.callback.on_epoch_begin.attach(cbk.on_epoch_begin)
        creator.callback.on_batch_begin.attach(cbk.on_batch_begin)
        creator.callback.on_batch_end.attach(cbk.on_batch_end)
        creator.callback.on_epoch_end.attach(cbk.on_epoch_end)
        creator.callback.on_train_end.attach(cbk.on_train_end)
    
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
        base_tag: str | None = None,
    ):
        self.args = args
        self.data = data
        self.generator = generator
        self.trainer = trainer
        self.callback = CustomCallback()

        base_tag = base_tag or f"{data.cache_key}@{time.strftime('%Y%m%d-%H%M%S')}"
        self.tag = pathlib.Path(*self.args.tags, base_tag)
        self.text_generator = TextGenerator(self.generator, tokenizer=self.data.tokenizer)

    def create_callbacks(self) -> t.Iterator[Callback]:
        self._attach_evaluators()
        yield ProgbarLogger(
            desc=str(self.tag),
            total=len(self.data.dataset['train']),
            updaters=self.trainer.updaters,
        )
        if self.args.tensorboard:
            self._attach_tensorboard()

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
            yield ModelCheckpoint(
                self.args,
                trainer=self.trainer,
                directory=self.args.checkpoint_root / self.tag,
                period=self.args.save_period,
            )
        else:
            warnings.warn("`checkpoint_root` is not given. Training can't be restored!")

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
        def calculate_mean_length(batch: int, _, channel=register_channel('samples')):
            if batch % 10 == 0:
                ids = self.text_generator.generate_ids(10)
                mean_length = np.mean(get_seqlens(ids, self.data.special_tokens.EOS.idx))
                channel.notify(batch, {'mean_length': mean_length})

        @self.callback.on_batch_end.attach
        def log_texts(batch: int, _):
            if batch % 100 == 0:
                sentenses = self.text_generator.generate_texts(3)
                print(SEPARATION_LINE)
                print()
                print(termcolor.colored("Real Sentences (Random Sampled):", 'blue'))
                print_samples(random_sample(self.data.dataset['train'].texts, len(sentenses)))
                print()
                print(termcolor.colored("Fake Sentences (Random Sampled):", 'red'))
                print_samples(sentenses)
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
                def bleu(batch: int, _, /, calculator=calculator, channel=register_channel(tag)):
                    if batch % 10 == 0:
                        ids = self.text_generator.generate_ids(ngram)
                        result = calculator.mean_bleu(ids)
                        channel.notify(batch, result)

            @self.callback.on_epoch_end.attach
            def self_bleu(epoch: int, channel=register_channel('samples')):
                ids = self.text_generator.generate_ids(len(self.data.dataset['train']))

                print("Evaluating generated data SelfBLEU...")
                print()
                selfbleu = BLEUCalculator.selfbleu(
                    ids,
                    max_gram=ngram,
                    eos_idx=self.data.special_tokens.EOS.idx,
                    smoothing=SmoothingFunction.fuzz_smoothing,
                )
                channel.notify(epoch, selfbleu)

        if fed_sample_size := self.args.fed:
            from core.evaluate import FEDCalculator

            for tag, dataset in self.data.dataset.items():
                print(f"Building '{tag}' data FED sentence encoder...")
                calculator = FEDCalculator(
                    hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                    references=random_sample(dataset.texts, size=fed_sample_size),
                )

                @self.callback.on_epoch_end.attach
                def fed(epoch: int, calculator=calculator, channel=register_channel(tag)):
                    texts = self.generator.generate_texts(fed_sample_size)
                    d = {'FED': calculator.calculate_fed_score(texts)}
                    channel.notify(epoch, d)

    def _attach_tensorboard(self):
        from tensorboardX import SummaryWriter

        logdir= self.args.tensorboard / self.tag
        writer = SummaryWriter(logdir=str(logdir))

        @self.callback.on_train_begin.attach
        def on_train_begin(is_restored: bool):
            logdir.mkdir(exist_ok=True)
            writer.add_text(
                'restore_args' if is_restored else 'args',
                ' '.join(sys.argv[1:]),
                0,
            )

        for updater in self.trainer.updaters:
            @updater.attach_subscriber
            def update_losses(step: int, losses: t.Mapping):
                if step % 10 == 0:
                    for key, val in losses.items():
                        writer.add_scalar(
                            tag=f'losses/{updater.module.scope}/{key}',
                            scalar_value=val,
                            global_step=step,  # TODO enerator step?
                        )

        for name, channel in CHANNELS.items():
            @channel.attach_subscriber
            def update_metrics(step: int, vals: t.Mapping):
                for key, val in vals.items():
                    writer.add_scalar(
                        tag=f'{name}/{key}',
                        scalar_value=val,
                        global_step=step,  # TODO batch ? epoch?
                    )


def print_samples(texts: t.Sequence[str]):
    for i, line in enumerate(texts, 1):
        print(f"{i}.")
        print(line)
