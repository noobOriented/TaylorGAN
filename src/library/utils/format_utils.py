import typing as t
from itertools import chain

import termcolor


def left_aligned(str_list: t.Iterable[str]) -> list[str]:
    maxlen = max(map(len, str_list), default=0)
    return [f"{s:<{maxlen}}" for s in str_list]


def format_path(path: object) -> str:
    return termcolor.colored(path, attrs=['underline'])


def format_id(id_str: str, bracket: bool = True) -> str:
    return termcolor.colored(f"[{id_str}]" if bracket else id_str, 'cyan')


def format_highlight(string: str, level: int = 0) -> str:
    if level == 0:
        bolder = "*" + "-" * (len(string) + 2) + "*"
        return termcolor.colored(
            f"{bolder}\n| {string.upper()} |\n{bolder}",
            color='yellow',
            attrs=['bold'],
        )
    return termcolor.colored(string, color='green')


def format_object(obj: object, /, *args: t.Any, **kwargs: t.Any):
    return f"{obj.__class__.__name__}({_join_arg_string(*args, **kwargs)})"


def _join_arg_string(*args, sep=', ', **kwargs):
    return sep.join(chain(
        map(str, args),
        (f"{k}={v}" for k, v in kwargs.items()),
    ))
