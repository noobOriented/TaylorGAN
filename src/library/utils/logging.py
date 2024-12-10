import builtins
import contextlib
import functools
import warnings
from unittest.mock import patch

import rich
import rich.text

from .format_utils import format_highlight


PRINT = builtins.print = rich.print  # guaranteed builtins!!
SEPARATION_LINE = rich.text.Text(' '.join('-' * 50), style='dark')
BULLETS = ["•", "–", "*", "·"]
_INDENT_LEVEL = 0


@contextlib.contextmanager
def logging_indent(header: str | None = None, bullet: bool = True):
    global _INDENT_LEVEL
    if header:
        if _INDENT_LEVEL == 0:
            print(format_highlight(header))
        elif _INDENT_LEVEL == 1:
            print(format_highlight(header, 1))
        else:
            print(header)

    contexts: list[contextlib.AbstractContextManager] = []
    if _INDENT_LEVEL == 0:  # need redirect
        if print != PRINT:
            warnings.warn("`logging_indent` should not be used with other redirector!")

        contexts += [
            patch('builtins.print', functools.partial(_print_body, bullet=bullet)),
            patch('rich.print', functools.partial(_print_body, bullet=bullet)),
        ]

    contexts.append(patch(f'{__name__}._INDENT_LEVEL', _INDENT_LEVEL + 1))
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield

    if _INDENT_LEVEL == 0:
        print(SEPARATION_LINE)
    elif _INDENT_LEVEL == 1:
        print()


def _print_body(*args, bullet: bool = True, **kwargs):
    if _INDENT_LEVEL < 2:
        PRINT(*args, **kwargs)
    elif bullet:
        bullet_symbol = BULLETS[min(_INDENT_LEVEL, len(BULLETS)) - 1]
        PRINT(' ' * (2 * _INDENT_LEVEL - 4) + bullet_symbol, *args, **kwargs)
    else:
        PRINT(' ' * (2 * _INDENT_LEVEL - 3), *args, **kwargs)
