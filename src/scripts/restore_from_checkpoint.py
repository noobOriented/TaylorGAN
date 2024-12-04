import os
import pathlib

import pydantic

from configs import GANTrainingConfigs, MLETrainingConfigs
from core.train.callbacks import ModelCheckpoint

from . import train


def main(args):
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


def parse_args(argv):
    from flexparse import SUPPRESS, ArgumentParser, IntRange

    from scripts.parsers import save_parser

    parser = ArgumentParser(
        parents=[save_parser(argument_default=SUPPRESS)],
        argument_default=SUPPRESS,
    )
    parser.add_argument(
        'path',
        type=pathlib.Path,
        help="load checkpoint from this file prefix.",
    )
    parser.add_argument(
        '--epochs',
        type=IntRange(1),
        help="number of training epochs. (default: same as original args.)",
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    import sys
    main(parse_args(sys.argv[1:]))
