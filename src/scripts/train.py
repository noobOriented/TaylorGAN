from configs import GANTrainingConfigs, MLETrainingConfigs
from core.train import DataLoader
from core.train.callbacks import ModelCheckpoint
from factories import callback_factory, data_factory, generator_factory, trainer_factory
from library.utils import logging_indent
from scripts.parsers import parse_args_as
from scripts.snippets import set_global_random_seed


def GAN_main():
    args = parse_args_as(GANTrainingConfigs)
    main(args)


def MLE_main():
    args = parse_args_as(MLETrainingConfigs)
    main(args)


def main(args: GANTrainingConfigs | MLETrainingConfigs, base_tag=None, checkpoint=None):
    with logging_indent("Set global random seed"):
        set_global_random_seed(args.random_seed)

    with logging_indent("Preprocess data"):
        data_collection, metadata = data_factory.preprocess(args)
        with logging_indent("Data summary:"):
            for key, array in data_collection.items():
                print(f"{key} data contains {len(array)} sentences.")

        metadata.tokenizer.summary()

    with logging_indent("Prepare Generator"):
        generator = args.create_generator(metadata)

    with logging_indent("Prepare Generator Trainer"):
        trainer = args.create_trainer(metadata, generator)
        trainer.summary()

    with logging_indent("Prepare Callback"):
        data_loader = DataLoader(
            data_collection['train'],
            batch_size=args.batch_size,
            n_epochs=args.epochs,
        )
        data_loader.callback = callback_factory.create(
            args,
            trainer=trainer,
            generator=generator,
            data_collection=data_collection,
            metadata=metadata,
            base_tag=base_tag,
        )

    if checkpoint:
        print(f"Restore from checkpoint: {checkpoint}")
        trainer.load_state(path=checkpoint)
        data_loader.skip_epochs(ModelCheckpoint.epoch_number(checkpoint))

    trainer.fit(data_loader)
