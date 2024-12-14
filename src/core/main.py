from __future__ import annotations

import os
import random
import typing as t

import numpy as np
import pydantic
import torch

from core.models import GeneratorConfigs
from core.train import CallbackConfigs, DataLoader, ModelCheckpointSaver, TrainerConfigs
from library.utils import format_highlight, logging_indent, parse_args_as
from preprocess import DataConfigs


def main(
    configs: MainConfigs | None = None,
    base_tag: str | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
):
    if configs is None:
        configs = parse_args_as(MainConfigs)

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


class MainConfigs(TrainerConfigs, CallbackConfigs, DataConfigs, GeneratorConfigs):
    epochs: t.Annotated[int, pydantic.Field(ge=1, description='number of training epochs.')] = 10_000
    batch_size: t.Annotated[int, pydantic.Field(ge=1, description='size of data mini-batch.')] = 64
    random_seed: t.Annotated[int | None, pydantic.Field(description='the global random seed.')] = None


if __name__ == '__main__':
    main()
