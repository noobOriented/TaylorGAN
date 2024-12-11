from __future__ import annotations

import collections
import dataclasses
import os
import pathlib
import sys
import time
import typing as t
import warnings

import numpy as np
import pydantic
import rich
import rich.live
import rich.panel
import rich.progress
import rich.table
import torch

from core.evaluate import TextGenerator
from core.models.generators import Generator
from core.preprocess import PreprocessResult
from core.train import Callback, EventHook, ModelCheckpointSaver, Trainer
from library.utils import (
    SEPARATION_LINE, ExponentialMovingAverageMeter, get_seqlens, logging_indent, random_sample,
)


class CallbackConfigs(pydantic.BaseModel):
    batch_size: t.Annotated[int, pydantic.Field(ge=1, description='size of data mini-batch.')] = 64

    # Evaluate
    bleu: t.Annotated[
        int | None,
        pydantic.Field(ge=1, le=5, description='longest n-gram to calculate BLEU/SelfBLEU score.'),
    ] = 5
    fed: t.Annotated[
        int | None,
        pydantic.Field(description='number of sample size for FED score.'),
    ] = None

    # Save
    checkpoint_root: pathlib.Path | None = None
    serving_root: pathlib.Path | None = None
    save_period: pydantic.PositiveInt = 1

    # Logging
    tensorboard: t.Annotated[
        pathlib.Path | None,
        pydantic.Field(description='whether to log experiment on tensorboard.')
    ] = None
    tags: t.Annotated[
        list[str],
        pydantic.Field(description='additional tags to configure this training (will be used in tensorboard).'),
    ] = []

    # Dev
    profile: pathlib.Path | None = None

    def get_callback(
        self,
        trainer: Trainer,
        generator: Generator,
        data: PreprocessResult,
        checkpoint: str | os.PathLike[str] | None = None,
        base_tag: str | None = None,
    ):
        creator = _CallbackCreator(
            self,
            data=data,
            generator=generator,
            trainer=trainer,
            checkpoint=checkpoint,
            base_tag=base_tag,
        )
        creator.attach_events()
        return creator.callback


@dataclasses.dataclass
class _CallbackCreator:
    args: CallbackConfigs
    generator: Generator
    trainer: Trainer
    data: PreprocessResult
    checkpoint: str | os.PathLike[str] | None
    base_tag: str | None = None
    
    def __post_init__(self):
        self.callback = Callback()
        self.tag = '/'.join([
            *self.args.tags,
            self.base_tag or f"{self.data.cache_key}@{time.strftime('%Y%m%d-%H%M%S')}",
        ])
        self.text_generator = TextGenerator(self.generator, tokenizer=self.data.tokenizer)
        self.metric_update_hooks = collections.defaultdict[str, EventHook[int, t.Any]](EventHook)

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

        @self.trainer.generator_updater.hook.attach
        def calculate_mean_length(batch: int, _, hook=self.metric_update_hooks['avg length']):
            if batch % 10 == 0:
                ids = self.text_generator.generate_ids(10)
                mean_length = np.mean(get_seqlens(ids, self.data.special_tokens.EOS.idx))
                hook(batch, float(mean_length))

        @self.trainer.generator_updater.hook.attach
        def log_texts(batch: int, _):
            if batch % 100 == 0:
                sentences = self.text_generator.generate_texts(3)
                print(SEPARATION_LINE)
                print()
                rich.print('[blue]Real Sentences (random sampled):')
                _print_samples(random_sample(self.data.dataset['train'].texts, len(sentences)))
                print()
                rich.print('[red]Fake Sentences (random sampled):')
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

                @self.trainer.generator_updater.hook.attach
                def compute_bleu(batch: int, _, c=calculator, hook=self.metric_update_hooks[f'{tag} BLEU 1~{ngram}']):
                    if batch % 10 == 0:
                        ids = self.text_generator.generate_ids(self.args.batch_size)
                        mean_bleu = c.bleu(ids).mean(0)
                        hook(batch, mean_bleu)

            @self.callback.on_epoch_end.attach
            def compute_selfbleu(epoch: int, hook=self.metric_update_hooks[f'self BLEU 1~{ngram}']):
                ids = self.text_generator.generate_ids(len(self.data.dataset['train']))
                mean_sbleu = BLEUCalculator.selfbleu(
                    ids,
                    max_gram=ngram,
                    eos_idx=self.data.special_tokens.EOS.idx,
                    smoothing=SmoothingFunction.fuzz_smoothing,
                ).mean(0)
                hook(epoch, mean_sbleu)

        if fed_sample_size := self.args.fed:
            from core.evaluate import FEDCalculator

            for tag, dataset in self.data.dataset.items():
                print(f"Building '{tag}' data FED sentence encoder...")
                calculator = FEDCalculator(
                    hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                    references=random_sample(dataset.texts, size=fed_sample_size),
                )

                @self.callback.on_epoch_end.attach
                def compute_fed(epoch: int, calculator=calculator, hook=self.metric_update_hooks[f'{tag}.FED']):
                    texts = self.generator.generate_texts(fed_sample_size)
                    score = calculator.calculate_fed_score(texts)
                    hook(epoch, score)

    def _attach_loggers(self):
        table = rich.table.Table.grid()
        table.add_row(str(self.tag), style='purple bold')
        table.add_row(
            self._create_modules_panel(),
            self._create_metric_panel(),
        )
        table.add_row(self._create_data_progress())

        live = rich.live.Live(table, refresh_per_second=10)
        self.callback.on_train_begin.attach(live.start)
        self.callback.on_train_end.attach(live.stop)

        if self.args.tensorboard:
            from tensorboardX import SummaryWriter

            logdir= self.args.tensorboard / self.tag
            writer = SummaryWriter(logdir=str(logdir))

            @self.callback.on_train_begin.attach
            def on_train_begin():
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

            for name, channel in self.metric_update_hooks.items():
                @channel.attach
                def update_metrics(step: int, val: float, tag=name.replace('.', '/')):
                    writer.add_scalar(
                        tag,
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

    def _create_modules_panel(self):

        def format_losses(losses: t.Mapping[str, float], /):
            return ' '.join(f'{k}={v:.3}' for k, v in losses.items())

        table = rich.table.Table.grid()

        for updater in self.trainer.updaters:
            progress = rich.progress.Progress('updates={task.completed}', '{task.fields[losses]}')
            task_id = progress.add_task('', losses='nan')
            ema = ExponentialMovingAverageMeter(decay=0.9)

            @updater.hook.attach
            def _(step, losses, progress=progress, task_id=task_id, ema=ema):
                losses = ema.apply(**losses)
                progress.update(task_id, advance=1, losses=_LazyFormatter(losses, format_losses))

            table.add_row(
                rich.panel.Panel(progress, border_style='blue', title=updater.module.scope, padding=(0, 2)),
            )

        return table

    def _create_metric_panel(self):
        progress = rich.progress.Progress('[cyan]{task.description}', '{task.fields[value]:.3}')

        def formatter(v, /):
            return np.array2string(np.asarray(v), precision=3).strip('[]')

        for name, hook in self.metric_update_hooks.items():
            task_id = progress.add_task(name, value='nan')
            @hook.attach
            def _(step, val, task_id=task_id):
                progress.update(task_id, value=_LazyFormatter(val, formatter))

        return rich.panel.Panel(progress, title='Metrics', border_style='cyan', padding=(1, 1))

    def _create_data_progress(self):
        progress = rich.progress.Progress(
            '[progress.description]{task.description}',
            rich.progress.SpinnerColumn(),
            rich.progress.BarColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TimeRemainingColumn(),
        )
        task_id = progress.add_task('', total=len(self.data.dataset['train']))

        @self.callback.on_epoch_begin.attach
        def reset_databar(epoch):
            progress.reset(task_id, description=f'Epoch {epoch}')

        @self.callback.on_batch_end.attach
        def update_databar(batch: int, batch_data):
            progress.update(task_id, advance=len(batch_data))

        return progress


@dataclasses.dataclass
class _LazyFormatter[T]:
    wrapped: T
    func: t.Callable[[T], str]

    def __format__(self, __format_spec):
        return self.func(self.wrapped)


def _print_samples(texts: t.Sequence[str], /):
    for i, line in enumerate(texts, 1):
        print(f"{i}.", line)
