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
import termcolor
import torch
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

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
        self.base_tag = self.base_tag or f"{self.data.cache_key}@{time.strftime('%Y%m%d-%H%M%S')}"
        self.tag = pathlib.Path(*self.args.tags, self.base_tag)
        self.text_generator = TextGenerator(self.generator, tokenizer=self.data.tokenizer)
        self.metric_channels = collections.defaultdict(EventHook[int, t.Mapping[str, float]])

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
        def calculate_mean_length(batch: int, _, hook=self.metric_channels['samples']):
            if batch % 10 == 0:
                ids = self.text_generator.generate_ids(10)
                mean_length = np.mean(get_seqlens(ids, self.data.special_tokens.EOS.idx))
                hook(batch, {'mean_length': float(mean_length)})

        @self.trainer.generator_updater.hook.attach
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

                @self.trainer.generator_updater.hook.attach
                def bleu(batch: int, _, /, calculator=calculator, hook=self.metric_channels[tag]):
                    if batch % 10 == 0:
                        ids = self.text_generator.generate_ids(self.args.batch_size)
                        result = calculator.mean_bleu(ids)
                        hook(batch, result)

            @self.callback.on_epoch_end.attach
            def self_bleu(epoch: int, hook=self.metric_channels['samples']):
                ids = self.text_generator.generate_ids(len(self.data.dataset['train']))
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
                def fed(epoch: int, calculator=calculator, hook=self.metric_channels[tag]):
                    texts = self.generator.generate_texts(fed_sample_size)
                    d = {'FED': calculator.calculate_fed_score(texts)}
                    hook(epoch, d)

    def _attach_loggers(self):
        loss_progress = _create_loss_progress({
            u.info: u.hook
            for u in self.trainer.updaters
        })
        metric_progress = _create_metric_progress(self.metric_channels)
        data_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        data_task = data_progress.add_task('', total=len(self.data.dataset['train']))

        table = Table(str(self.tag))
        table.add_row(Panel.fit(loss_progress, title='Losses', border_style="green"))
        table.add_row(Panel(metric_progress, title='Metrics', border_style="cyan"))
        table.add_row(data_progress)
        live = Live(table, refresh_per_second=10)

        self.callback.on_train_begin.attach(live.start)
        self.callback.on_train_end.attach(live.stop)

        @self.callback.on_epoch_begin.attach
        def reset_databar(epoch):
            data_progress.reset(data_task, description=f'Epoch {epoch}')

        @self.callback.on_batch_end.attach
        def update_databar(batch: int, batch_data):
            data_progress.update(data_task, advance=len(batch_data))

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

            for name, channel in self.metric_channels.items():
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


def _create_loss_progress(
    module_update_hooks: t.Mapping[str, EventHook[int, t.Mapping]],
):
    progress = Progress(
        '[green]{task.description}[/green]',
        '{task.completed}',
        '{task.fields[values]}',
    )
    for name, hook in module_update_hooks.items():
        @hook.attach
        def update_losses(
            step, losses,
            task_id=progress.add_task(name, values=''),
            ema_meter=ExponentialMovingAverageMeter(decay=0.9),
        ):
            losses = ema_meter.apply(**losses)
            progress.update(task_id, advance=1, values=_NumericalDict(**losses))

    return progress


def _create_metric_progress(
    metric_update_hooks: t.Mapping[str, EventHook[int, t.Mapping]],
):
    progress = Progress('[cyan]{task.description}[/cyan]', '{task.fields[values]}', expand=True)

    for name, hook in metric_update_hooks.items():
        @hook.attach
        def update_metrics(
            step, vals,
            task_id=progress.add_task(name, values=''),
            ema_meter=ExponentialMovingAverageMeter(decay=0.)  # to persist logged values,
        ):
            vals = ema_meter.apply(**vals)
            progress.update(task_id, values=_NumericalDict(**vals))

    return progress


class _NumericalDict(dict[str, float]):

    def __init__(self, **kwargs: float) -> None:
        super().__init__(kwargs)
    
    def __format__(self, __format_spec):
        return ', '.join(f'{k}={v:.3}' for k, v in self.items())



def _print_samples(texts: t.Sequence[str], /):
    for i, line in enumerate(texts, 1):
        print(f"{i}.", line)
