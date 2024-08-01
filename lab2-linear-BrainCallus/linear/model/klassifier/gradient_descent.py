from typing import Callable

import numpy as np
from model.klassifier.linear_klassifier import LinearKlassifier
from model.params.loss_function import LossFunction


class GradientDescentClassifier(LinearKlassifier):
    def __init__(self, learning_rate=0.01, num_iters=100, alpha=0.1, beta=0.1,
                 loss_function: LossFunction = None,
                 pred_transformer: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        super().__init__(pred_transformer)
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.alpha = alpha
        self.beta = beta
        self.loss_function = loss_function
        self.weights = None

    def regularize(self, grad):
        grad -= (self.beta * self.weights + self.alpha * np.sign(self.weights))
        return grad

    def fit(self, train_exdog, train_ans, curve_func: Callable[[np.ndarray, np.ndarray], float] = None):
        if curve_func is None:
            self.logger.warn(
                f'Loss function is not defined, used default for this instance {self.loss_function.common_name}')
            curve_func = self.loss_function.calc_loss

        self.train_exdog = train_exdog.copy()
        self.train_ans = train_ans.copy()
        n_samples, n_features = train_exdog.shape
        self.weights = np.zeros(n_features)
        errors = []
        for _ in range(self.num_iters):
            preds = self.activation_func(np.dot(train_exdog, self.weights))
            grad = self.loss_function.calc_grad(x=train_exdog, y_expect=train_ans, y_real=preds)

            grad = self.regularize(grad)
            self.weights -= self.learning_rate * grad
            errors.append(curve_func(train_ans, preds))

        return errors

    def _internal_predict(self, test_exdog):
        preds = self.activation_func(np.dot(test_exdog, self.weights))
        return np.where(preds > 0.0, 1, 0)

    @staticmethod
    def activation_func(x):
        return 1 / (1 + np.exp(x))

    def calc_loss(self, exdog, y):
        return self.loss_function.calc_loss(np.matmul(exdog, self.weights), y)
