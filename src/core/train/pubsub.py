from __future__ import annotations

import typing as t


class ListenableEvent[*T]:

    def __init__(self) -> None:
        self._hooks: list[t.Callable[[*T], t.Any]] = []

    def __call__(self, *args: *T) -> None:
        for h in self._hooks:
            h(*args)

    def register_hook(self, f: t.Callable[[*T], t.Any], /):
        self._hooks.append(f)
        return f
