
from __future__ import annotations

import typing as t

import numpy as np
import torch

from core.losses import GeneratorLoss
from core.models import Generator, TokenSequence
from core.train import GeneratorTrainer, ModuleUpdater
from library.utils import cache_method_call

from ._discriminator import Discriminator, DiscriminatorLoss


class GANTrainer(GeneratorTrainer):

    def __init__(
        self,
        generator: Generator,
        optimizer: torch.optim.Optimizer,
        losses: t.Mapping[str, tuple[GeneratorLoss, float]],
        discriminator: Discriminator,
        discriminator_optimizer: torch.optim.Optimizer,
        discriminator_losses: t.Mapping[str, tuple[DiscriminatorLoss, float]],
        d_steps: int = 1,
    ):
        super().__init__(generator, optimizer, losses)
        self._discriminator = discriminator
        self._d_steps = d_steps
        self._discriminator_updater = ModuleUpdater(
            discriminator,
            discriminator_optimizer,
            discriminator_losses,
        )
        self.loss_update_events['Discriminator'] = self._discriminator_updater.loss_update_events

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=self._generator.special_tokens.EOS.idx,
            )
            with (
                cache_method_call(self._generator, 'generate'),
                cache_method_call(self._discriminator, 'score_samples'),
                cache_method_call(self._discriminator, 'score_word_vector'),
                cache_method_call(self._discriminator, 'get_embedding'),
            ):
                fake_samples = self._generator.generate(real_samples.batch_size, real_samples.maxlen)
                self._discriminator_updater.update_step(real_samples, fake_samples)
                if self._discriminator_updater.step % self._d_steps == 0:
                    self._generator_updater.update_step(real_samples)

    @property
    def _updaters(self):
        return super()._updaters + [self._discriminator_updater]
