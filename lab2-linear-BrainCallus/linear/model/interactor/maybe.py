from typing import Callable


class Maybe[T]:
    def __init__(self, value: T = None):
        self.value = value

    def map[R](self, mapper: Callable[[T], R]):
        return Maybe(None if self.value is None else mapper(self.value))

    def get(self) -> T:
        if self.value is None:
            raise ValueError('Value is None')
        return self.value

    def get_or_else(self, default: T) -> T:
        return default if self.value is None else self.value
