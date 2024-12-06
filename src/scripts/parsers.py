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
            if field.default is False:
                parser.add_argument(flag, action='store_true', help=field.description)
            else:
                parser.add_argument('--no-' + flag[2:], action='store_false', help=field.description)

            continue

        arg_type = t.get_args(field.annotation)[0] if field.default is None else field.annotation
        parser.add_argument(
            flag,
            required=field.is_required(),
            nargs='+' if lenient_issubclass(t.get_origin(arg_type), list) else None,
            default=field.default,
            help=field.description,
        )

    namespace = parser.parse_args(args)
    return typ.model_validate(namespace, from_attributes=True)


def lenient_issubclass(cls, base: tuple[type] | type) -> bool:
    return isinstance(cls, type) and issubclass(cls, base)


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
