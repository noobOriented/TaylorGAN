from __future__ import annotations

import typing as t

from tqdm import tqdm

from core.train.updaters import ModuleUpdater
from library.utils import ExponentialMovingAverageMeter, format_highlight2, left_aligned
from library.utils.logging import SEPARATION_LINE, TqdmRedirector

from .base import Callback
from .channels import channels


class _Bar(t.Protocol):
    def close(self):
        ...


class ProgbarLogger(Callback):

    def __init__(self, desc: str, total: int, updaters: t.Sequence[ModuleUpdater]):
        self.desc = format_highlight2(desc)
        self.total = total
        self._updaters = updaters
        self._bars: list[_Bar] = []

    def on_train_begin(self, is_restored: bool):
        TqdmRedirector.enable()
        self._add_bar(bar_format=SEPARATION_LINE)
        self.header = self._add_bar(
            bar_format="{desc}: {elapsed}",
            desc=self.desc,
        )
        for updater in self._updaters:
            updater.attach_subscriber(self._add_bar(_ModuleBar, desc=updater.info))

        self._add_bar(bar_format=SEPARATION_LINE)

        for channel, m_aligned in zip(channels.values(), left_aligned(channels.keys())):
            channel.attach_subscriber(self._add_bar(_MetricsBar, desc=m_aligned))

        self._add_bar(bar_format=SEPARATION_LINE)

    def _add_bar[T: _Bar](self, bar_cls: type[T] = tqdm, **kwargs) -> T:
        bar = bar_cls(
            file=TqdmRedirector.STDOUT,  # use original stdout port
            dynamic_ncols=True,
            position=-len(self._bars),
            **kwargs,
        )
        self._bars.append(bar)
        return bar

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


class _MetricsBar:

    def __init__(self, desc: str, **kwargs):
        self.pbar = self._PostfixBar(desc=desc, unit="step", **kwargs)
        self.ema_meter = ExponentialMovingAverageMeter(decay=0.)  # to persist logged values

    def update(self, step, vals):
        smoothed_vals = self.ema_meter.apply(**vals)
        self.pbar.set_postfix(smoothed_vals)

    def close(self):
        self.pbar.close()

    class _PostfixBar(tqdm):

        # HACK override: remove the leading `,` of postfix
        # https://github.com/tqdm/tqdm/blob/master/tqdm/_tqdm.py#L255-L457
        @staticmethod
        def format_meter(
                n, total, elapsed, ncols=None, prefix='', ascii=False,  # noqa: A002
                unit='it', unit_scale=False, rate=None, bar_format=None,
                postfix=None, unit_divisor=1000, **extra_kwargs,
            ):
            if prefix:
                prefix = prefix + ': '
            if not postfix:
                postfix = 'nan'
            return f"{prefix}{postfix}"


class _ModuleBar:

    def __init__(self, desc: str, **kwargs):
        self.pbar = self._ModuleBar(desc=desc, **kwargs)
        self.ema_meter = ExponentialMovingAverageMeter(decay=0.9)

    def update(self, step, losses):
        if step > self.pbar.n:
            self.pbar.update(step - self.pbar.n)

        smoothed_losses = self.ema_meter.apply(**losses)
        self.pbar.set_postfix(smoothed_losses)

    def close(self):
        self.pbar.close()

    class _ModuleBar(tqdm):

        # HACK override: remove the leading `,` of postfix
        # https://github.com/tqdm/tqdm/blob/master/tqdm/_tqdm.py#L255-L457
        @staticmethod
        def format_meter(
            n, total, elapsed, ncols=None, prefix='', ascii=False,  # noqa: A002
            unit='it', unit_scale=False, rate=None, bar_format=None,
            postfix=None, unit_divisor=1000, **extra_kwargs,
        ):
            if not postfix:
                return f"{prefix} steps: {n}"
            return f"{prefix} steps: {n}, losses: [{postfix}]"
