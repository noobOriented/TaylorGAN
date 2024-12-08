import os

from core.train import DataLoader
from core.train.callbacks import ModelCheckpoint
from factories import callback_factory
from library.utils import logging_indent, parse_args_as
from scripts.snippets import set_global_random_seed

from ._configs import GANTrainingConfigs, MLETrainingConfigs


def GAN_main():
    configs = parse_args_as(GANTrainingConfigs)
    main(configs)


def MLE_main():
    configs = parse_args_as(MLETrainingConfigs)
    main(configs)


def main(
    configs: GANTrainingConfigs | MLETrainingConfigs,
    base_tag=None,
    checkpoint: str | os.PathLike[str] | None = None,
):
    with logging_indent("Set global random seed"):
        set_global_random_seed(configs.random_seed)

    with logging_indent("Preprocess data"):
        preprocessed_result = configs.load_data()
        preprocessed_result.summary()

    with logging_indent("Prepare Generator"):
        generator = configs.get_generator(preprocessed_result)

    with logging_indent("Prepare Generator Trainer"):
        trainer = configs.get_trainer(preprocessed_result, generator)
        trainer.summary()

    data_loader = DataLoader(
        dataset=preprocessed_result.dataset['train'].ids,
        batch_size=configs.batch_size,
        n_epochs=configs.epochs,
    )
    with logging_indent("Prepare Callback"):
        callback_factory.create(
            configs,
            data=preprocessed_result,
            data_loader=data_loader,
            generator=generator,
            trainer=trainer,
            base_tag=base_tag,
        )

    if checkpoint:
        print(f"Restore from checkpoint: {checkpoint}")
        trainer.load_state(path=checkpoint)
        data_loader.skip_epochs(ModelCheckpoint.epoch_number(checkpoint))

    trainer.fit(data_loader)
