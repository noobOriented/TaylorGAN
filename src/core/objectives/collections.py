import typing as t


class LossCollection:

    def __init__(self, total, **observables):
        self.total = total
        self.observables = observables

    def __radd__(self, other) -> t.Self:
        return self + other

    def __add__(self, other) -> t.Self:
        if isinstance(other, LossCollection):
            return LossCollection(
                self.total + other.total,
                **self.observables,
                **other.observables,
            )
        if other == 0:
            return LossCollection(self.total + 0, **self.observables)

        raise TypeError
