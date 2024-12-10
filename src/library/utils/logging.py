import builtins
import sys
import warnings
from contextlib import contextmanager
from functools import partial

import termcolor

from .format_utils import format_highlight


STDOUT, STDERR, PRINT = sys.stdout, sys.stderr, builtins.print  # guaranteed builtins!!
SEPARATION_LINE = termcolor.colored(' '.join(['-'] * 50), attrs=['dark'])


@contextmanager
def logging_indent(header: str = None, bullet: bool = True):
    if header:
        _IndentPrinter.print_header(header)

    if _IndentPrinter.level == 0:  # need redirect
        if builtins.print != PRINT:
            warnings.warn("`logging_indent` should not be used with other redirector!")
        builtins.print = partial(_IndentPrinter.print_body, bullet=bullet)

    _IndentPrinter.level += 1
    yield
    _IndentPrinter.level -= 1

    if _IndentPrinter.level == 0:  # need recover
        builtins.print = PRINT
    _IndentPrinter.print_footer()


class _IndentPrinter:

    '''DO NOT use it with TqdmRedirector!!!'''

    BULLETS = ["•", "–", "*", "·"]
    level = 0

    @classmethod
    def print_header(cls, header):
        if cls.level == 0:
            print(format_highlight(header))
        elif cls.level == 1:
            print(format_highlight(header, 1))
        else:
            print(header)

    @classmethod
    def print_body(cls, *args, bullet: bool = True, **kwargs):
        assert cls.level > 0
        if cls.level < 2:
            PRINT(*args, **kwargs)
        elif bullet:
            bullet_symbol = cls.BULLETS[min(cls.level, len(cls.BULLETS)) - 1]
            PRINT(' ' * (2 * cls.level - 2) + bullet_symbol, *args, **kwargs)
        else:
            PRINT(' ' * (2 * cls.level - 3), *args, **kwargs)

    @classmethod
    def print_footer(cls):
        if cls.level == 0:
            print(SEPARATION_LINE)
        elif cls.level == 1:
            print()

