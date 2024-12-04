import os
import pathlib

import pydantic
import argparse

from configs import GANTrainingConfigs, MLETrainingConfigs
from core.train.callbacks import ModelCheckpoint
from scripts.parsers import save_parser

from . import train


def main():
    parser = argparse.ArgumentParser(
        parents=[save_parser(argument_default=argparse.SUPPRESS)],
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        'path',
        type=pathlib.Path,
        help="load checkpoint from this file prefix.",
    )
    parser.add_argument(
        '--epochs',
        type=int,  # IntRange(1),
        help="number of training epochs. (default: same as original args.)",
    )
    args = parser.parse_args()
    restore_path = args.path
    main_args_path = restore_path / 'args'
    try:
        with open(main_args_path, 'r') as f_in:
            main_args: GANTrainingConfigs | MLETrainingConfigs = pydantic.TypeAdapter(
                GANTrainingConfigs | MLETrainingConfigs,
            ).validate_json(f_in.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"{main_args_path} not found, checkpoint can't be restored.")

    main_args.__dict__.update(args.__dict__)
    train.main(
        main_args,
        base_tag=os.path.basename(restore_path),
        checkpoint=ModelCheckpoint.latest_checkpoint(restore_path),
    )


if __name__ == '__main__':
    main()
