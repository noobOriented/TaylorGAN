from core.train.pubsub_base import Subject


channels: dict[str, Subject] = {}


def register_channel(key: str) -> Subject:
    return channels.setdefault(key, Subject())
