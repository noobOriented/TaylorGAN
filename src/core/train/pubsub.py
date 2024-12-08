from __future__ import annotations

import typing as t


CHANNELS: dict[str, Channel] = {}


def register_channel(key: str) -> Channel:
    return CHANNELS.setdefault(key, Channel())


class Subscriber(t.Protocol):

    def __call__(self, step: int, vals: t.Mapping[str, float], /):
        ...


class Channel:

    def __init__(self):
        self._subscribers: list[Subscriber] = []

    def attach_subscriber[T: Subscriber](self, subcriber: T, /) -> T:
        self._subscribers.append(subcriber)
        return subcriber

    def notify(self, step: int, vals: t.Mapping[str, float]):
        for s in self._subscribers:
            s(step, vals)
