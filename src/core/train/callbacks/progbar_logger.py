from __future__ import annotations

import typing as t

import tqdm

from core.train.updaters import ModuleUpdater
from library.utils import (
    SEPARATION_LINE, ExponentialMovingAverageMeter, TqdmRedirector, format_highlight2, left_aligned,
)

from ..pubsub import CHANNELS
from .base import Callback


class ProgbarLogger(Callback):

    def __init__(self, desc: str, total: int, updaters: t.Sequence[ModuleUpdater]):
        self.desc = format_highlight2(desc)
        self.total = total
        self._updaters = updaters
        self._bars: list[tqdm.tqdm] = []

    def on_train_begin(self, is_restored: bool):
        TqdmRedirector.enable()
        self._add_bar(bar_format=SEPARATION_LINE)
        self.header = self._add_bar(
            bar_format="{desc}: {elapsed}",
            desc=self.desc,
        )
        for updater in self._updaters:
            pbar = self._add_bar(desc=updater.info)
            pbar.format_meter = _format_meter_for_losses
            ema_meter = ExponentialMovingAverageMeter(decay=0.9)

            @updater.attach_subscriber
            def update_losses(step, losses, pbar=pbar, ema_meter=ema_meter):
                if step > pbar.n:
                    pbar.update(step - pbar.n)
                pbar.set_postfix(ema_meter.apply(**losses))

        self._add_bar(bar_format=SEPARATION_LINE)

        for channel, m_aligned in zip(CHANNELS.values(), left_aligned(CHANNELS.keys())):
            pbar = self._add_bar(desc=m_aligned)
            pbar.format_meter = _format_meter_for_metrics
            ema_meter = ExponentialMovingAverageMeter(decay=0.)  # to persist logged values

            @channel.attach_subscriber
            def update_metrics(step, vals, pbar=pbar, ema_meter=ema_meter):
                pbar.set_postfix(ema_meter.apply(**vals))

        self._add_bar(bar_format=SEPARATION_LINE)

    def on_epoch_begin(self, epoch):
        self.body = self._add_bar(
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            desc=f"Epoch {epoch}",
            total=self.total,
            unit='sample',
            unit_scale=True,
            leave=False,
        )

    def on_batch_end(self, batch: int, batch_data):
        self.header.refresh()
        self.body.update(len(batch_data))

    def on_epoch_end(self, epoch):
        self._bars.pop()
        self.body.close()

    def on_train_end(self):
        for bar in self._bars:
            bar.close()
        TqdmRedirector.disable()

    def _add_bar(self, **kwargs):
        bar = tqdm.tqdm(
            file=TqdmRedirector.STDOUT,  # use original stdout port
            dynamic_ncols=True,
            position=-len(self._bars),
            **kwargs,
        )
        self._bars.append(bar)
        return bar


# HACK override: remove the leading `,` of postfix
# https://github.com/tqdm/tqdm/blob/master/tqdm/_tqdm.py#L255-L457
def _format_meter_for_metrics(*, prefix='', postfix=None, **kwargs):
    if prefix:
        prefix = prefix + ': '
    if not postfix:
        postfix = 'nan'
    return f"{prefix}{postfix}"


def _format_meter_for_losses(*, n, prefix='', postfix=None, **kwargs):
    if not postfix:
        return f"{prefix} steps: {n}"
    return f"{prefix} steps: {n}, losses: [{postfix}]"
