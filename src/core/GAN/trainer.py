
from __future__ import annotations

import typing as t

import numpy as np
import torch

from core.losses import GeneratorLoss
from core.models import Generator
from core.models.sequence_modeling import TokenSequence
from core.train import ListenableEvent, ModuleTrainingState, Trainer
from library.utils import cache_method_call

from .discriminators import Discriminator, DiscriminatorLoss


class GANTrainer(Trainer):

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

        self.discriminator = discriminator
        self.discriminator_losses = discriminator_losses
        self.d_steps = d_steps
        self.loss_update_events[self.discriminator.scope] = {
            k: ListenableEvent()
            for k in discriminator_losses.keys()
        }

        self._discriminator_state = ModuleTrainingState(discriminator, discriminator_optimizer)

    def fit(self, data_loader: t.Iterable[np.ndarray], /):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=self._generator_state.module.special_tokens.EOS.idx,
            )
            with (
                cache_method_call(self.generator, 'generate'),
                cache_method_call(self.discriminator, 'score_samples'),
                cache_method_call(self.discriminator, 'score_word_vector'),
                cache_method_call(self.discriminator, 'get_embedding'),
            ):
                sum_loss = self._compute_discriminator_loss(real_samples)
                self._discriminator_state.update_step(sum_loss)
                if self._discriminator_state.step % self.d_steps == 0:
                    sum_loss = self._compute_generator_loss(real_samples)
                    self._generator_state.update_step(sum_loss)

    @property
    def _module_states(self):
        return super()._module_states + [self._discriminator_state]

    def _compute_discriminator_loss(self, real_samples: TokenSequence) -> torch.Tensor:
        fake_samples = self.generator.generate(real_samples.batch_size, real_samples.maxlen)
        losses = {
            name: loss_fn(self.discriminator, real_samples, fake_samples)
            for name, (loss_fn, _) in self.discriminator_losses.items()
        }
        for name, loss_val in losses.items():
            self.loss_update_events[self.discriminator.scope][name](
                self._discriminator_state.step,
                loss_val.detach().numpy(),
            )

        return sum(self.discriminator_losses[k][1] * v for k, v in losses.items())  # type: ignore
