from __future__ import annotations

import typing as t


METRIC_CHANNELS: dict[str, EventHook[int, t.Mapping[str, float]]] = {}


def register_channel(key: str) -> EventHook[int, t.Mapping[str, float]]:
    return METRIC_CHANNELS.setdefault(key, EventHook())


class EventHook[*T]:

    def __init__(self) -> None:
        self._hooks: list[t.Callable[[*T], t.Any]] = []

    def __call__(self, *args: *T) -> None:
        for f in self._hooks:
            f(*args)

    def attach(self, f: t.Callable[[*T], t.Any], /):
        self._hooks.append(f)
        return f
