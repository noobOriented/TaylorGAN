
from __future__ import annotations

import typing as t

import numpy as np
import torch

from core.models.sequence_modeling import TokenSequence
from core.train import GeneratorUpdater, ModuleUpdater, Trainer
from library.utils import cache_method_call

from .discriminators import Discriminator


class GANTrainer(Trainer):

    def __init__(
        self,
        generator_updater: GeneratorUpdater,
        discriminator_updater: DiscriminatorUpdater,
        d_steps: int = 1,
    ):
        super().__init__(generator_updater)
        self.discriminator_updater = discriminator_updater
        self.d_steps = d_steps

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=1,
            )
            self.discriminator_updater.update_step(
                real_samples=real_samples,
                fake_samples=self.generator_updater.module.generate(*batch_data.shape),
            )
            if self.discriminator_updater.step % self.d_steps == 0:
                self.generator_updater.update_step(real_samples)

    @property
    def updaters(self):
        return super().updaters + [self.discriminator_updater]


class DiscriminatorUpdater(ModuleUpdater[Discriminator]):

    def compute_loss(self, real_samples, fake_samples):
        with (
            cache_method_call(self.module, 'score_samples'),
            cache_method_call(self.module, 'score_word_vector'),
            cache_method_call(self.module, 'get_embedding'),
        ):
            return {
                name: loss(discriminator=self.module, real_samples=real_samples, fake_samples=fake_samples)
                for name, (loss, _) in self.losses.items()
            }
