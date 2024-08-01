import math
from abc import ABC, abstractmethod
from enum import Enum


class KType(Enum):
    UNIFORM = 'Uniform'
    GAUSSIAN = 'Gaussian'
    TRIANGULAR = 'Triangular'
    EPANECHIKOV = 'Epanechnikov'


class Kernel(ABC):
    def __init__(self, name: KType):
        self.name = name

    @property
    def name(self):
        return self.name

    @abstractmethod
    def eval(self, x: float) -> float:
        pass

    @name.setter
    def name(self, value):
        self._name = value


class UniformKernel(Kernel):
    def __init__(self):
        super().__init__(KType.UNIFORM)

    def eval(self, x: float) -> float:
        return 0.5 if -1 < x < 1 else 0


class GaussKernel(Kernel):
    def __init__(self):
        super().__init__(KType.GAUSSIAN)

    def eval(self, x: float) -> float:
        return 1 / math.sqrt(2 * math.pi) * math.exp(-(x ** 2 / 2))


class TriangKernel(Kernel):
    def __init__(self):
        super().__init__(KType.UNIFORM)

    def eval(self, x: float) -> float:
        return max(1 - abs(x), 0)


class EpanechKernel(Kernel):
    def __init__(self):
        super().__init__(KType.UNIFORM)

    def eval(self, x: float) -> float:
        return max(3 / 4 * (1 - x ** 2), 0)
