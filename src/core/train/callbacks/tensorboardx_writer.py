import pathlib
import sys
import typing as t

from tensorboardX import SummaryWriter

from core.train.updaters import ModuleUpdater
from library.utils import format_path

from ..pubsub import CHANNELS
from .base import Callback


class TensorBoardXWritter(Callback):

    def __init__(self, logdir: pathlib.Path, log_period: int, updaters: t.Sequence[ModuleUpdater]):
        self.logdir = logdir
        self.log_period = log_period
        self.updaters = updaters
        self.writer = SummaryWriter(logdir=str(self.logdir))

    def on_train_begin(self, is_restored: bool):
        self.logdir.mkdir(exist_ok=True)
        self.writer.add_text(
            'restore_args' if is_restored else 'args',
            ' '.join(sys.argv[1:]),
            0,
        )
        for updater in self.updaters:

            @updater.attach_subscriber
            def update_losses(step: int, losses: t.Mapping):
                if step % self.log_period == 0:
                    for key, val in losses.items():
                        self.writer.add_scalar(
                            tag=f'losses/{updater.module.scope}/{key}',
                            scalar_value=val,
                            global_step=step,  # TODO enerator step?
                        )

        for name, channel in CHANNELS.items():
            @channel.attach_subscriber
            def update_metrics(step: int, vals: t.Mapping):
                for key, val in vals.items():
                    self.writer.add_scalar(
                        tag=f'{name}/{key}',
                        scalar_value=val,
                        global_step=step,  # TODO batch ? epoch?
                    )

    def get_config(self):
        return {'period': self.log_period, 'logdir': format_path(self.logdir)}
