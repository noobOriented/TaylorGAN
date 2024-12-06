import argparse
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
