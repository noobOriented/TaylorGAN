import abc
import numbers
import typing as t


class Subscriber(t.Protocol):

    @abc.abstractmethod
    def update(self, step: int, vals: t.Mapping[str, numbers.Number], /):
        ...


class Subject:

    def __init__(self):
        self._subscribers: list[Subscriber] = []

    def attach_subscriber(self, subcriber: Subscriber):
        self._subscribers.append(subcriber)

    def notify(self, step, losses):
        for s in self._subscribers:
            s.update(step, losses)
