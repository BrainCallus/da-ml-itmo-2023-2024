from abc import ABC, abstractmethod
from enum import Enum
from math import sqrt


class WType(Enum):
    RELATIVE = 'Relative'
    FIXED = 'Fixed'


class Window(ABC):
    def __init__(self, name: WType):
        self.name = name

    @property
    def name(self):
        return self.name

    @name.setter
    def name(self, value):
        self._name = value

    @abstractmethod
    def get_neighbors_cnt(self, data) -> int:
        pass

    @abstractmethod
    def get_kernel_param_divvisor(self, data):
        pass


class FixedWindow(Window):
    def __init__(self, h: float):
        super().__init__(WType.FIXED)
        self.h = h

    def get_neighbors_cnt(self, data) -> int:
        return min(int(sqrt(len(data))), len(data) - 1)

    def get_kernel_param_divvisor(self, data):
        return self.h


class RelativeWindow(Window):
    def __init__(self, k: int):
        super().__init__(WType.RELATIVE)
        self.k = k

    def get_neighbors_cnt(self, data) -> int:
        return self.k + 1

    def get_kernel_param_divvisor(self, data):
        return data[-1]
