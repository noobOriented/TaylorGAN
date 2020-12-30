import inspect
from functools import wraps, WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, update_wrapper
from typing import List

from .collections import dict_of_unique
from .format_utils import format_list


def match_abbrev(func):
    func_args = get_args(func)
    bypass = inspect.getfullargspec(func).varkw is not None

    def match_abbrev(abbrev):
        matches = [kw for kw in func_args if kw.startswith(abbrev)]
        if len(matches) > 1:
            raise TypeError(
                f"ambiguous: {abbrev} match multiple results: {format_list(matches)}",
            )
        if len(matches) == 1:
            return matches[0]
        elif bypass:  # too short
            return abbrev

        raise TypeError(
            f"{func.__qualname__} got an unexpected keyword argument {repr(abbrev)}, "
            f"allowed arguments of {func.__qualname__}: {format_list(func_args)}",
        )

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            new_kwargs = dict_of_unique(
                (match_abbrev(key), val) for key, val in kwargs.items()
            )
        except ValueError as e:
            raise TypeError(f"more than one abbrev match to the same keyword: {e}")

        return func(*args, **new_kwargs)

    return wrapped


def get_args(func) -> List[str]:
    func_args = inspect.getfullargspec(func).args
    if func_args and func_args[0] in ('self', 'cls'):
        return func_args[1:]
    return func_args


class ObjectWrapper:

    def __init__(self, body):
        self._body = body

    def __getattr__(self, name):
        return getattr(self._body, name)


class ArgumentBinder:

    def __init__(self, func, preserved=()):
        old_sig = inspect.signature(func)
        preserved = set(preserved)
        self.func = func
        self.__signature__ = old_sig.replace(
            parameters=[
                param
                for key, param in old_sig.parameters.items()
                if param.name not in preserved
            ],
        )

    def __call__(self, *b_args, **b_kwargs):
        binding = self.__signature__.bind_partial(*b_args, **b_kwargs)

        def bound_function(*args, **kwargs):
            return self.func(
                *args,
                **kwargs,
                **binding.arguments,
            )

        return bound_function


def wraps_with_new_signature(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):

    def update_wrapper_signature(wrapper):
        wrapper = update_wrapper(wrapper, wrapped=wrapped, assigned=assigned, updated=updated)
        old_sig = inspect.signature(wrapped)
        add_params = [
            p
            for p in inspect.signature(wrapper, follow_wrapped=False).parameters.values()
            if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ]
        new_sig = old_sig.replace(parameters=[*old_sig.parameters.values(), *add_params])
        wrapper.__signature__ = new_sig
        return wrapper

    return update_wrapper_signature
