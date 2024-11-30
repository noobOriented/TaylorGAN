from numbers import Number

from core.train.pubsub_base import Subject


class MessageChannel(Subject):

    def post(self, step: int, vals: dict[str, Number]):
        for subscriber in self._subscribers:
            subscriber.update(step, vals)


channels: dict[str, MessageChannel] = {}


def register_channel(key: str):
    return channels.setdefault(key, MessageChannel())
