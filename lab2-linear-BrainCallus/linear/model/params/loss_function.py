from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class FunType(Enum):
    MSE = 'squared_error'
    BCE = 'log_loss'
    HINGE = 'hinge'


class LossFunction(ABC):
    def __init__(self, lib_name: FunType, common_name: str = None):
        self.name = lib_name
        self.common_name = lib_name if common_name is None else common_name

    @abstractmethod
    def calc_loss(self, y_expect, y_real):
        pass

    def calc_grad(self, x, y_expect, y_real):
        pass


class Mse(LossFunction):
    def __init__(self):
        super().__init__(FunType.MSE, 'mse')

    def calc_loss(self, y_expect, y_real):
        return np.mean((y_real - y_expect) ** 2)

    def calc_grad(self, x, y_expect, y_real):
        return 2 * np.dot(x.T, (y_real - y_expect)) / x.shape[0]


class Bce(LossFunction):
    def __init__(self):
        super().__init__(FunType.BCE, 'binary_cross_entropy')
        self.eps = 1e-9

    def calc_loss(self, y_expect, y_real):
        y_exp = np.clip(np.array(y_expect), self.eps, 1 - self.eps)
        return -1 * np.mean(y_real * np.log(y_exp) + (1 - y_real) * np.log(1 - y_exp))

    def calc_grad(self, x, y_expect, y_real):
        y_expect_ = np.clip(np.array(y_expect), self.eps, 1 - self.eps)
        y_real_ = np.clip(y_real, self.eps, 1 - self.eps)
        return -1 * np.dot(x.T, (y_real_ / y_expect_ - (1 - y_real_) / (1 - y_expect_))) / x.shape[0]


class HingeLoss(LossFunction):
    def __init__(self):
        super().__init__(FunType.HINGE)

    def calc_loss(self, y_expect, y_real):
        return max(0, np.average(1 - y_expect * y_real))

    def calc_grad(self, x, y_expect, y_real):
        if self.calc_loss(y_expect, y_real) > 0:
            return np.dot(x.T, -y_expect)
        else:
            return np.dot(x.T, np.zeros(len(y_expect)))


LOSS_FUNCTION_MAPPING = dict(zip([f_type.value for f_type in FunType], LossFunction.__subclasses__()))
