from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import scipy.spatial.distance as dist


class KType(Enum):
    POLYNOMIAL = 'Polynomial'
    LINEAR = 'Linear'
    GAUSSIAN = 'Gaussian'


class Kernel(ABC):
    def __init__(self, k_type: KType):
        self.k_type = k_type

    @abstractmethod
    def compute(self, x, y) -> np.ndarray:
        pass


class PolynomialKernel(Kernel):
    def __init__(self, degree: float, k_type=KType.POLYNOMIAL):
        super().__init__(k_type)
        self.degree = degree

    def compute(self, x, y):
        return np.dot(x, y.T) ** self.degree


class LinearKernel(PolynomialKernel):
    def __init__(self):
        super().__init__(degree=1, k_type=KType.LINEAR)


class GaussianKernel(Kernel):
    def __init__(self, gamma: float = 0.1):
        super().__init__(KType.GAUSSIAN)
        self.gamma = gamma

    def compute(self, x, y):
        return np.exp(-self.gamma * dist.cdist(np.atleast_2d(x), np.atleast_2d(y)) ** 2).flatten()


KERNEL_MAPPING = dict(zip([k_type.value for k_type in KType], Kernel.__subclasses__()))
