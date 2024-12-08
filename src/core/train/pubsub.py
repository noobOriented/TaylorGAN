from __future__ import annotations

import abc
import typing as t


CHANNELS: dict[str, Channel] = {}


def register_channel(key: str) -> Channel:
    return CHANNELS.setdefault(key, Channel())


class Subscriber(t.Protocol):

    @abc.abstractmethod
    def update(self, step: int, vals: t.Mapping[str, float], /):
        ...


class Channel:

    def __init__(self):
        self._subscribers: list[Subscriber] = []

    def attach_subscriber(self, subcriber: Subscriber, /):
        self._subscribers.append(subcriber)

    def notify(self, step: int, vals: t.Mapping[str, float]):
        for s in self._subscribers:
            s.update(step, vals)
