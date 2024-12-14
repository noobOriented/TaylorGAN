import argparse
import pathlib

import pydantic

import core.main
from core.GAN import GANMainConfigs
from core.train import ModelCheckpointSaver


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        'path',
        type=pathlib.Path,
        help="load checkpoint from this file prefix.",
    )
    # TODO pydantic Model?
    parser.add_argument(
        '--checkpoint-root', '--ckpt',
        type=pathlib.Path,
        help="save checkpoint to this directory.",
    )
    parser.add_argument(
        '--serving-root',
        type=pathlib.Path,
        help='save serving model to this directory.',
    )
    parser.add_argument(
        '--save-period',
        type=int,  # IntRange(minval=1),
        default=1,
        help="interval (number of epochs) between each saving.",
    )
    parser.add_argument(
        '--epochs',
        type=int,  # IntRange(1),
        help="number of training epochs. (default: same as original args.)",
    )
    args = parser.parse_args()

    restore_path: pathlib.Path = args.path
    main_args_path = restore_path / 'args'
    try:
        with open(main_args_path, 'r') as f_in:
            main_args = pydantic.RootModel[
                GANMainConfigs | core.main.MainConfigs,
            ].model_validate_json(f_in.read()).root
    except FileNotFoundError:
        raise FileNotFoundError(f"{main_args_path} not found, checkpoint can't be restored.")

    main_args.__dict__.update(args.__dict__)
    core.main.main(
        main_args,
        base_tag=restore_path.name,
        checkpoint=ModelCheckpointSaver.latest_checkpoint(restore_path),
    )


if __name__ == '__main__':
    main()
