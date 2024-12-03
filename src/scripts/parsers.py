import os
import pathlib

from dotenv import load_dotenv
from flexparse import ArgumentParser, IntRange

from library.utils import format_path


load_dotenv('.env')


def evaluate_parser(**kwargs):
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('evaluate', description='Settings of evaluation metrics.')
    group.add_argument(
        '--bleu',
        nargs='?',
        type=IntRange(1, 5),
        const=5,
        help="longest n-gram to calculate BLEU/SelfBLEU score (5 if not specified).",
    )
    group.add_argument(
        '--fed',
        nargs='?',
        type=IntRange(minval=1),
        const=10000,
        help="number of sample size for FED score.",
    )
    return parser


def save_parser(**kwargs):
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('save', description="Settings of saving model.")
    group.add_argument(
        '--checkpoint-root', '--ckpt',
        type=pathlib.Path,
        help="save checkpoint to this directory.",
    )
    if group.get_default('checkpoint_root') is None:  # to avoid interfering SUPPRESS
        group.set_defaults(checkpoint_root=os.getenv('CHECKPOINT_DIR'))

    group.add_argument(
        '--serving-root',
        type=pathlib.Path,
        help='save serving model to this directory.',
    )
    group.add_argument(
        '--save-period',
        type=IntRange(minval=1),
        default=1,
        help="interval (number of epochs) between each saving.",
    )
    return parser


def load_parser(**kwargs):
    parser = ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('load', description='Settings of loading saved model.')
    group.add_argument(
        '--model-path',
        type=pathlib.Path,
        required=True,
        help='path of serving model folder.',
    )
    return parser


def develop_parser():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group('develop', description='Developers only options.')
    group.add_argument(
        '--profile',
        nargs='?',
        type=pathlib.Path,
        const='./profile_stats',
        help=f"export profile stats to file ({format_path('./profile_stats')} if not specified).",
    )
    return parser