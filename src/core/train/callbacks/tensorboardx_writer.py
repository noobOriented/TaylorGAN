import os
import pathlib
import sys
import typing as t

from tensorboardX import SummaryWriter

from library.utils import format_path

from ..pubsub import CHANNELS
from .base import Callback


class TensorBoardXWritter(Callback):

    def __init__(self, logdir: pathlib.Path, log_period: int, updaters):
        self.logdir = logdir
        self.log_period = log_period
        self.updaters = updaters

    def on_train_begin(self, is_restored: bool):
        self.logdir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(logdir=str(self.logdir))
        self.writer.add_text(
            'restore_args' if is_restored else 'args',
            ' '.join(sys.argv[1:]),
            0,
        )
        for updater in self.updaters:
            updater.attach_subscriber(
                ModuleLossWriter(
                    self.writer,
                    tag_template=os.path.join('losses', updater.module.scope, '{key}'),
                    log_period=self.log_period,
                ),
            )

        for name, channel in CHANNELS.items():
            channel.attach_subscriber(MetricsWriter(
                self.writer,
                tag_template=os.path.join(name, '{key}'),
            ))

    def get_config(self):
        return {'period': self.log_period, 'logdir': format_path(self.logdir)}


class ModuleLossWriter:

    def __init__(self, writer, tag_template, log_period: int = 1):
        self.writer = writer
        self.tag_template = tag_template
        self.log_period = log_period

    def update(self, step: int, vals: t.Mapping):
        if step % self.log_period == 0:
            for key, val in vals.items():
                self.writer.add_scalar(
                    tag=self.tag_template.format(key=key),
                    scalar_value=val,
                    global_step=step,  # TODO enerator step?
                )


class MetricsWriter:

    def __init__(self, writer, tag_template):
        self.writer = writer
        self.tag_template = tag_template

    def update(self, step: int, vals: t.Mapping):
        for key, val in vals.items():
            self.writer.add_scalar(
                tag=self.tag_template.format(key=key),
                scalar_value=val,
                global_step=step,  # TODO batch ? epoch?
            )
