import typing as t
from functools import reduce


def counter_or(dicts: t.Iterable[dict[t.Any, int]]):
    return reduce(_counter_ior, dicts, {})


def _counter_ior(a: dict[t.Any, int], b: dict[t.Any, int]):
    # NOTE much faster than Counter() |
    for key, cnt in b.items():
        a[key] = max(cnt, a.get(key, cnt))
    return a


class ExponentialMovingAverageMeter:

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.prev_val: float | None = None

    def __call__(self, new_val: float, /) -> float:
        if self.prev_val is None:
            self.prev_val = new_val
        else:
            self.prev_val = self.prev_val * self.decay + new_val * (1. - self.decay)

        return self.prev_val
