from __future__ import annotations

import os
import random
import typing as t

import numpy as np
import pydantic
import torch

from core.models import GeneratorConfigs
from core.preprocess import DataConfigs
from library.utils import format_highlight, logging_indent, parse_args_as

from ._callback_factory import CallbackConfigs
from ._fit_loop import DataLoader
from ._loss import GeneratorLoss, mean_negative_log_likelihood
from ._trainer import GeneratorTrainer, ModelCheckpointSaver
from ._trainer_factory import _G_REGS, _OPTIMIZERS, MLEObjectiveConfigs


def MLE_main():
    configs = parse_args_as(MLEConfigs)
    main(configs)


def main(
    configs: CommonTrainingConfigs,
    base_tag: str | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
):
    with logging_indent("Set global random seed"):
        _set_global_random_seed(configs.random_seed)

    with logging_indent("Preprocess data"):
        preprocessed_result = configs.load_data()
        preprocessed_result.summary()

    with logging_indent("Prepare Generator"):
        generator = configs.get_generator(preprocessed_result)

    with logging_indent("Prepare Generator Trainer"):
        trainer = configs.get_trainer(preprocessed_result, generator)
        trainer.summary()

    with logging_indent("Prepare Callback"):
        callback = configs.get_callback(
            data=preprocessed_result,
            generator=generator,
            trainer=trainer,
            checkpoint=checkpoint,
            base_tag=base_tag,
        )
        callback.summary()

    data_loader = DataLoader(
        dataset=preprocessed_result.dataset['train'].ids,
        batch_size=configs.batch_size,
        n_epochs=configs.epochs,
        callback=callback,
    )
    if checkpoint:
        print(f"Restore from checkpoint: {checkpoint}")
        trainer.load_state(path=checkpoint)
        data_loader.skip_epochs(ModelCheckpointSaver.epoch_number(checkpoint))

    print(format_highlight("Start Training"))
    trainer.fit(data_loader)


def _set_global_random_seed(seed: int | None):
    print(f"seed = {seed}")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class CommonTrainingConfigs(CallbackConfigs, DataConfigs, GeneratorConfigs):
    epochs: t.Annotated[int, pydantic.Field(ge=1, description='number of training epochs.')] = 10_000
    batch_size: t.Annotated[int, pydantic.Field(ge=1, description='size of data mini-batch.')] = 64
    random_seed: t.Annotated[int | None, pydantic.Field(description='the global random seed.')] = None


class MLEConfigs(CommonTrainingConfigs, MLEObjectiveConfigs, extra='forbid'):

    def get_trainer(self, data, generator):
        losses: dict[str, tuple[GeneratorLoss, float]] = {'NLL': (mean_negative_log_likelihood, 1)}
        for s in self.g_regularizers:
            (reg, coeff), info = _G_REGS(s, return_info=True)
            losses[info.func_name] = (reg, coeff)

        return GeneratorTrainer(
            generator,
            optimizer=_OPTIMIZERS(self.g_optimizer)(generator.parameters()),
            losses=losses,
        )
