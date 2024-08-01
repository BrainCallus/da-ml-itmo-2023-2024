from typing import Callable

import numpy as np
from model.klassifier.linear_klassifier import LinearKlassifier


class RidgeRegression(LinearKlassifier):
    def __init__(self, alpha: float, pred_transformer=lambda x: x):
        super().__init__(pred_transformer)
        self.alpha = alpha
        self.coef = None

    def fit(self, exdog, train_ans, curve_func: Callable[[np.ndarray, np.ndarray], float] = None):
        n_samples, n_features = exdog.shape
        self.train_exdog = np.column_stack((np.ones(n_samples), exdog))

        a = np.dot(self.train_exdog.T, self.train_exdog) + self.alpha * np.eye(n_features + 1)
        b = np.dot(self.train_exdog.T, train_ans)
        self.coef = np.linalg.solve(a, b)

        return self

    def _internal_predict(self, test_exdog):
        n_samples, _ = test_exdog.shape
        return np.dot(np.column_stack((np.ones(n_samples), test_exdog)), self.coef)

    def regularization(self, grad):
        return grad - self.alpha * self.coef
