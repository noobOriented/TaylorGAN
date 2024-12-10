from __future__ import annotations

import typing as t

from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from library.utils import ExponentialMovingAverageMeter

from .pubsub import EventHook


class ProgbarLogger:

    def __init__(
        self,
        desc: str,
        total: int,
        module_update_hooks: t.Mapping[str, EventHook[int, t.Mapping]],
        metric_update_hooks: t.Mapping[str, EventHook[int, t.Mapping]],
    ):
        loss_progress = Progress('[green]{task.description}[/green]', '{task.fields[values]}', expand=True)
        metric_progress = Progress('[cyan]{task.description}[/cyan]', '{task.fields[values]}', expand=True)
        self._data_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        self._databar_task = self._data_progress.add_task('', total=total)

        for name, hook in module_update_hooks.items():
            task_id = loss_progress.add_task(name, values='')
            ema_meter = ExponentialMovingAverageMeter(decay=0.9)

            @hook.attach
            def update_losses(step, losses, task_id=task_id, ema_meter=ema_meter):
                losses = ema_meter.apply(**losses)
                loss_progress.update(task_id, values=_NumericalDict(**losses))

        for name, hook in metric_update_hooks.items():
            task_id = metric_progress.add_task(name, values='')
            ema_meter = ExponentialMovingAverageMeter(decay=0.)  # to persist logged values

            @hook.attach
            def update_metrics(step, vals, task_id=task_id, ema_meter=ema_meter):
                vals = ema_meter.apply(**vals)
                metric_progress.update(task_id, values=_NumericalDict(**vals))

        self.table = Table(desc)
        self.table.add_row(
            Panel.fit(loss_progress, title='Losses', border_style="green"),
        )
        self.table.add_row(
            Panel(metric_progress, title='Metrics', border_style="cyan"),
        )
        self.table.add_row(
            self._data_progress,
        )

    def on_train_begin(self):
        self.live = Live(self.table)
        self.live.start()

    def on_train_end(self):
        self.live.stop()

    def on_epoch_begin(self, epoch):
        self._data_progress.update(self._databar_task, description=f'Epoch {epoch}', completed=0)

    def on_batch_end(self, batch: int, batch_data):
        self._data_progress.update(self._databar_task, advance=len(batch_data))


class _NumericalDict(dict[str, float]):

    def __init__(self, **kwargs: float) -> None:
        super().__init__(kwargs)
    
    def __format__(self, __format_spec):
        return ', '.join(f'{k}={v:.3}' for k, v in self.items())
