from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from model.log.logger import Logger


class LinearKlassifier(ABC):
    def __init__(self, pred_transformer: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        self.train_exdog = None
        self.train_ans = None
        self.pred_transformer = pred_transformer
        self.logger = Logger(self.__class__.__name__)

    @abstractmethod
    def fit(self, train_exdog, train_endog, curve_func: Callable[[np.ndarray, np.ndarray], float] = None):
        pass

    def predict(self, test_exdog):
        return self.pred_transformer(self._internal_predict(test_exdog))

    def raw_predict(self, test_exdog):
        return self._internal_predict(test_exdog)

    @abstractmethod
    def _internal_predict(self, test_exdog):
        pass
