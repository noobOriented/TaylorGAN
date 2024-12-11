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


class _IndentState:
    level: int = 0


@contextlib.contextmanager
def logging_indent(header: str | None = None, bullet: bool = True):
    if header:
        if _IndentState.level == 0:
            print(format_highlight(header))
        elif _IndentState.level == 1:
            print(format_highlight(header, 1))
        else:
            print(header)

    contexts: list[contextlib.AbstractContextManager] = []
    if _IndentState.level == 0:  # need redirect
        if print != PRINT:
            warnings.warn("`logging_indent` should not be used with other redirector!")

        contexts += [
            patch.object(builtins, 'print', functools.partial(_print_body, bullet=bullet)),
            patch.object(rich, 'print', functools.partial(_print_body, bullet=bullet)),
        ]

    contexts.append(patch.object(_IndentState, 'level', _IndentState.level + 1))
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield

    if _IndentState.level == 0:
        print(SEPARATION_LINE)
    elif _IndentState.level == 1:
        print()


def _print_body(*args, bullet: bool = True, **kwargs):
    level = _IndentState.level
    if level < 2:
        PRINT(*args, **kwargs)
    elif bullet:
        bullet_symbol = BULLETS[min(level, len(BULLETS)) - 1]
        PRINT(' ' * (2 * level - 4) + bullet_symbol, *args, **kwargs)
    else:
        PRINT(' ' * (2 * level - 3), *args, **kwargs)
