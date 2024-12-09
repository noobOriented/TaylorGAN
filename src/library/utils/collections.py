import typing as t
from functools import reduce


def counter_or(dicts: t.Iterable[dict[t.Any, int]]):
    return reduce(counter_ior, dicts, {})


def counter_ior(a: dict[t.Any, int], b: dict[t.Any, int]):
    # NOTE much faster than Counter() |
    for key, cnt in b.items():
        if cnt > a.get(key, 0):
            a[key] = cnt
    return a


class ExponentialMovingAverageMeter:

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.prev_vals: dict[str, float] = {}

    def apply(self, **kwargs: float) -> dict[str, float]:
        for key, val in kwargs.items():
            new_val = self.prev_vals.get(key, val) * self.decay + val * (1. - self.decay)
            self.prev_vals[key] = new_val

        return self.prev_vals
