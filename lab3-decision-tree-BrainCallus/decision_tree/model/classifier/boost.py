from typing import Tuple

import numpy as np

from logger.logger import Logger


class AdaBoostSamme:
    def __init__(self, base_estimator, n_estimators: int = 50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.logger = Logger(AdaBoostSamme.__name__)
        self.n_classes_: int = -1
        self.classes_: np.ndarray = np.empty(0)
        self.estimator_weights: [float] = None
        self.estimators: list = []

    def fit(self, train_exdog, train_ans):
        train_exdog = np.array(train_exdog)
        n_samples, _n_features = self._reset_before_fit(train_exdog, train_ans)
        weights = np.ones(n_samples) / n_samples
        mod = int(max(10, self.n_estimators // 10))

        for i in range(self.n_estimators):
            weights = self._process_estimator(train_exdog, train_ans, weights)
            if i % mod == 0:
                self.logger.info(f'Processed {i} / {self.n_estimators} estimators; ' +
                                 f'last weight: {self.estimator_weights[-1]}')

        self.logger.info(f'Processed {self.n_estimators} / {self.n_estimators} estimators')

    def _reset_before_fit(self, train_exdog: np.ndarray, train_ans: np.ndarray) -> Tuple[int, int]:
        self.classes_ = np.arange(np.max(train_ans) + 1)
        self.n_classes_ = len(self.classes_)
        self.estimator_weights = []
        self.estimators = []
        ns, nf = train_exdog.shape
        return ns, nf

    def _process_estimator(self, train_exdog: np.ndarray, train_ans: np.ndarray, weights: np.ndarray) -> np.ndarray:
        estimator = self.base_estimator()
        estimator.fit(train_exdog, train_ans, weights=weights)
        y_pred = estimator.predict(train_exdog)
        incorrect = y_pred != train_ans
        # alpha - the vote weight of the weak classifier
        alpha = self._calc_estimator_weight(weights, incorrect)

        self.estimator_weights.append(alpha)
        self.estimators.append(estimator)
        weights *= np.exp(alpha * incorrect * ((weights > 0) | (alpha < 0)))
        return weights / np.sum(weights)

    def _calc_estimator_weight(self, weights: np.ndarray, incorrect) -> float:
        error = np.dot(weights, incorrect) / np.sum(weights)
        return np.log((1 - error) / (error + 1e-9)) + np.log(self.n_classes_ - 1)

    # noinspection PyUnresolvedReferences
    def predict(self, test_exdog):
        classes = self.classes_[:, np.newaxis]
        pred = sum(
            np.where((estimator.predict(test_exdog) == classes).T, alpha, -1 / (self.n_classes_ - 1) * alpha)
            for estimator, alpha in zip(self.estimators, self.estimator_weights)
        ) / np.array(self.estimator_weights).sum()

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
