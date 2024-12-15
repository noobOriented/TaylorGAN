from __future__ import annotations

import argparse
import ast
import inspect
import re
import typing as t

import pydantic


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
            nargs='+' if _lenient_issubclass(t.get_origin(arg_type), list) else None,
            default=field.default,
            help=field.description,
        )

    namespace = parser.parse_args(args)
    return typ.model_validate(namespace, from_attributes=True)


def _lenient_issubclass(cls, base: tuple[type] | type) -> bool:
    return isinstance(cls, type) and issubclass(cls, base)


class LookUpCall[T]:

    def __init__(self, choices: t.Mapping[str, t.Callable[..., T]]):
        self.choices = choices

    def parse(self, s: str | T) -> T:
        return self(s) if isinstance(s, str) else s

    @t.overload
    def __call__(self, arg_string: str, return_info: t.Literal[True]) -> tuple[T, ArgumentInfo]:
        ...

    @t.overload
    def __call__(self, arg_string: str, return_info: t.Literal[False] = False) -> T:
        ...

    def __call__(self, arg_string: str, return_info: bool = False):
        # clean color
        ANSI_CLEANER = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]")
        arg_string = ANSI_CLEANER.sub("", arg_string)
        func_name, pos_args, kwargs = _get_func_name_and_args(arg_string)
        func = self.choices[func_name]
        result = func(*pos_args, **kwargs)
        if return_info:
            info = ArgumentInfo(arg_string, func_name, func, pos_args, kwargs)
            return result, info
        return result

    def get_helps(self):
        for key, func in self.choices.items():
            yield f'{key}{inspect.signature(func)}'


class ArgumentInfo(t.NamedTuple):
    arg_string: str
    func_name: str
    func: t.Callable
    args: list
    kwargs: dict


def _get_func_name_and_args(string: str) -> tuple[str, list, dict]:
    node = ast.parse(string, mode='eval').body
    if isinstance(node, ast.Name):
        return node.id, [], {}
    if not isinstance(node, ast.Call):
        raise ValueError("can't be parsed as a call.")

    return (
        node.func.id,
        [ast.literal_eval(arg) for arg in node.args],
        dict_of_unique_keys(
            (kw.arg, ast.literal_eval(kw.value))
            for kw in node.keywords
        ),
    )


def dict_of_unique_keys(items):
    output = {}
    for key, val in items:
        if key in output:
            raise ValueError(f"keyword argument repeated: {key}")
        output[key] = val
    return output
