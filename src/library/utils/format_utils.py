import itertools
import typing as t

import rich.text


def format_highlight(string: str, level: int = 0):
    if level == 0:
        bolder = "*" + "-" * (len(string) + 2) + "*"
        return rich.text.Text(f"{bolder}\n| {string.upper()} |\n{bolder}", style='bold yellow')
    return rich.text.Text(string, style='green')


def format_object(obj: object, /, *args: t.Any, **kwargs: t.Any):
    return f"{obj.__class__.__name__}({_join_arg_string(*args, **kwargs)})"


def _join_arg_string(*args, sep=', ', **kwargs):
    return sep.join(itertools.chain(
        map(str, args),
        (f"{k}={v}" for k, v in kwargs.items()),
    ))
