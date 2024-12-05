import argparse
import os
import pathlib
import typing as t

import pydantic
from dotenv import load_dotenv


load_dotenv('.env')


def parse_args_as[T: pydantic.BaseModel](typ: type[T], *, args: t.Sequence[str] | None = None) -> T:
    parser = argparse.ArgumentParser()
    for name, field in typ.model_fields.items():
        # TODO group/multiple flags
        flag = '--' + name.replace('_', '-')
        if field.annotation is bool:
            parser.add_argument(
                flag,
                action='store_true' if field.default is False else 'store_false',
                help=field.description,
            )
            continue

        if field.default is None:
            arg_type=  t.get_args(field.annotation)[0]
        else:
            arg_type = field.annotation

        if lenient_issubclass(t.get_origin(arg_type), list):
            arg_type = t.get_args(arg_type)[0]
            nargs = '+'
        else:
            nargs = None

        parser.add_argument(
            flag,
            required=field.is_required(),
            nargs=nargs,
            type=arg_type,
            default=field.default,
            help=field.description,
        )

    namespace = parser.parse_args(args)
    return typ.model_validate(namespace, from_attributes=True)


def lenient_issubclass(cls, base: tuple[type] | type) -> bool:
    return isinstance(cls, type) and issubclass(cls, base)


def evaluate_parser(**kwargs):
    parser = argparse.ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('evaluate', description='Settings of evaluation metrics.')
    group.add_argument(
        '--bleu',
        nargs='?',
        type=int,  # IntRange(1, 5),
        const=5,
        help="longest n-gram to calculate BLEU/SelfBLEU score (5 if not specified).",
    )
    group.add_argument(
        '--fed',
        nargs='?',
        type=int,  # IntRange(minval=1),
        const=10000,
        help="number of sample size for FED score.",
    )
    return parser


def save_parser(**kwargs):
    parser = argparse.ArgumentParser(add_help=False, **kwargs)
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
        type=int,  # IntRange(minval=1),
        default=1,
        help="interval (number of epochs) between each saving.",
    )
    return parser


def load_parser(**kwargs):
    parser = argparse.ArgumentParser(add_help=False, **kwargs)
    group = parser.add_argument_group('load', description='Settings of loading saved model.')
    group.add_argument(
        '--model-path',
        type=pathlib.Path,
        required=True,
        help='path of serving model folder.',
    )
    return parser
