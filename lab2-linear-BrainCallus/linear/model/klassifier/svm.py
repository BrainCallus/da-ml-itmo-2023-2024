from typing import Callable

import numpy as np
from model.log.logger import Logger
from model.klassifier.linear_klassifier import LinearKlassifier


class SVM(LinearKlassifier):
    def __init__(self, c: float, kernel, tolerance: float, num_iters: int = 1000, kernel_matrix=None,
                 pred_transformer: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        super().__init__(pred_transformer)
        self.c = c
        self.kernel = kernel
        self.tolerance = tolerance
        self.num_iters = num_iters
        self.kernel_matrix = kernel_matrix
        self.alphas = None
        self.supported_indices = []
        self.b = 0.0
        self.logger = Logger(SVM.__name__)

    def initialize(self):
        if self.kernel_matrix is None:
            self._init_kernel_matrix()
        self.alphas = np.zeros(len(self.train_exdog))
        self.supported_indices = np.arange(0, len(self.train_exdog))

    def fit(self, exdog, endog, curve_func: Callable[[np.ndarray, np.ndarray], float] = None):
        if curve_func is None:
            self.logger.warn('No loss function passed, can\'t build learning curve')

        self.train_exdog = np.array(exdog.copy())
        self.train_ans = np.array(endog.copy())
        self.initialize()
        errs = []

        for i in range(self.num_iters):
            if i % 25 == 0:
                self.logger.info(f'Processed {i}/{self.num_iters} iterations')

            alphas_prev = np.copy(self.alphas)
            for j in range(len(self.train_exdog)):
                idx = self._rand_idx_exclude_value(j)
                self.make_step(idx, j)

            if curve_func is not None:
                # loss_fun(y_expect, y_real) !not permute
                errs.append(curve_func(self.pred_transformer(self.train_ans), self.predict(self.train_exdog)))

            if np.linalg.norm(self.alphas - alphas_prev) < self.tolerance:
                self.logger.info(f'Reasonable tolerance {self.tolerance} was achieved on iteration {i}')
                break

        self.supported_indices = np.where(self.alphas > 0)[0]
        return errs

    def make_step(self, idx1, idx2):
        self.logger.linebreak = False
        self.logger.linebreak = True
        eta = 2 * self.kernel_matrix[idx1][idx2] - self.kernel_matrix[idx1][idx2] - self.kernel_matrix[idx2][idx2]
        if eta >= 0:
            return

        err1 = self.calc_err(idx1)
        err2 = self.calc_err(idx2)
        alpha1, alpha2 = self.calc_alphas(idx1, idx2, err1, err2)
        self.b = self.calc_b(alpha1, alpha2, idx1, idx2, err1, err2)

    def _internal_predict(self, test_exdog):
        x = np.array(test_exdog.copy())
        return np.sign([self.calc_row(x_i) for x_i in x])

    def calc_row(self, x):
        k_v = self.kernel.compute(self.train_exdog[self.supported_indices], x)
        return np.dot((self.alphas[self.supported_indices] * self.train_ans[self.supported_indices]).T, k_v.T) + self.b

    def calc_err(self, idx):
        return self.calc_row(self.train_exdog[idx]) - self.train_ans[idx]

    def calc_b(self, alpha1, alpha2, idx1, idx2, err1, err2):
        beta1 = self._calc_beta(alpha1, alpha2, idx1, idx2, err1, self.kernel_matrix[idx1][idx2])
        beta2 = self._calc_beta(alpha2, alpha1, idx2, idx1, err2, self.kernel_matrix[idx1][idx2])
        if 0 < self.alphas[idx1] < self.c:
            return beta1
        elif 0 < self.alphas[idx2] < self.c:
            return beta2
        else:
            return (beta1 + beta2) / 2

    def _calc_beta(self, a1, a2, idx1, idx2, err, m):
        return (self.b - err -
                self.train_ans[idx1] * (self.alphas[idx1] - a1) * self.kernel_matrix[idx1][idx1] -
                self.train_ans[idx2] * (self.alphas[idx2] - a2) * m)

    def calc_alphas(self, idx1, idx2, err1, err2):
        low, high = self._get_bounds(idx1, idx2)
        alpha1 = self.alphas[idx1]
        alpha2 = self.alphas[idx2]

        eta = 2 * self.kernel_matrix[idx1][idx2] - self.kernel_matrix[idx1][idx2] - self.kernel_matrix[idx2][idx2]
        self.alphas[idx2] -= (self.train_ans[idx2] * (err1 - err2)) / eta
        self.alphas[idx2] = self.clip_alpha(self.alphas[idx2], low, high)

        eps = 1e-8
        if self.alphas[idx2] < eps:
            self.alphas[idx2] = 0
        elif self.alphas[idx2] > (self.c - eps):
            self.alphas[idx2] = self.c

        self.alphas[idx1] += self.train_ans[idx1] * self.train_ans[idx2] * (alpha2 - self.alphas[idx2])
        return alpha1, alpha2

    @staticmethod
    def clip_alpha(alpha, low, high):
        if low <= alpha <= high:
            return alpha
        elif alpha < low:
            return low
        elif alpha > high:
            return high

    def _get_bounds(self, idx1, idx2):
        if self.train_ans[idx1] != self.train_ans[idx2]:
            return (
                max(0, self.alphas[idx2] - self.alphas[idx1]),
                min(self.c, self.c - self.alphas[idx1] + self.alphas[idx2]))

        else:
            return (
                max(0, self.alphas[idx1] + self.alphas[idx2] - self.c),
                min(self.c, self.alphas[idx1] + self.alphas[idx2]))

    def _init_kernel_matrix(self) -> None:
        self.kernel_matrix = np.zeros((len(self.train_exdog), len(self.train_exdog)))
        for i in range(len(self.train_exdog)):
            self.kernel_matrix[:, i] = self.kernel.compute(self.train_exdog, self.train_exdog[i, :])

    def _rand_idx_exclude_value(self, value: int) -> int:
        res = value
        while res == value:
            res = np.random.randint(0, len(self.train_exdog) - 1)

        return res
