from typing import TypeVar, Callable

import numpy as np
from optuna import Trial, trial
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from model.params.Kernel import *
from model.params.Metric import *
from model.params.Window import *
from model.knnClassifier import KNNClassifier
import optuna
from logger.logger import Logger

KERNEL_MAPPING = dict(zip([k_type.value for k_type in KType], Kernel.__subclasses__()))


class KnnInteractor:
    def __init__(self, exdog: DataFrame, endog: [int], test_part: float = 0.2):
        self.logger = Logger(KnnInteractor.__name__)
        self.exdog = exdog
        self.endog = endog
        self.train_exdog, self.test_exdog, self.train_ans, self.test_ans = \
            train_test_split(self.exdog, self.endog, test_size=test_part, shuffle=False)
        self.default_weights = self.get_fair_weights(self.train_exdog)
        self.best_knn_params_impl: (Window, Kernel, Metric) = None, None, None  # (window with param, kernel, metric)
        self.best_knn_params_lib: (int, str, str, float) = None, None, None  # (k, weight func,  algorithm, p(mink deg))

    @property
    def best_knn_params_impl(self):
        return self._best_knn_params_impl

    @best_knn_params_impl.setter
    def best_knn_params_impl(self, value: (Window, Kernel, Metric)):
        self._best_knn_params_impl = value

    def get_best_impl_klassifier(self):
        best_fixed = self._run_study(self._objective_for_fixed)
        best_relative = self._run_study(self._objective_for_relative)
        self.logger.info(f'Best fixed: {best_fixed}')
        self.logger.info(f'Best relative: {best_relative}')
        if best_fixed.values[0] > best_relative.values[0]:
            params = best_fixed.params
            window = FixedWindow(params.get('fixed:h'))
            self.logger.info('FIXED better')
            self.logger.info(f'Best params[FIXED]: {best_fixed.params}')
        else:
            params = best_relative.params
            window = RelativeWindow(params.get('relative:k'))
            self.logger.info('RELATIVE better')
            self.logger.info(f'Best impl params[RELATIVE]: {best_relative.params}')
        self.best_knn_params_impl = window, self.get_kernel_by_enum(params.get('kernel')), params.get('metric')

    def get_best_lib_klassifier(self):
        best_lib = self._run_study(self._objective_lib)
        self.logger.info(f'Best scikit: {best_lib}')
        params = best_lib.params
        self.best_knn_params_lib = params.get('k'), params.get('weight function'), params.get('algorithm'), params.get(
            'minkowski degree')
        self.logger.info(f'Best scikit params: {params}')

    def get_accuracy_weighted(self, x: DataFrame, y: int, w: float, anomalies1=None, anomalies2=None) -> float:
        if anomalies1 is None or anomalies2 is None:
            _, anomalies1, anomalies2 = self.get_anomalies(self.train_exdog, self.train_ans, self.best_knn_params_impl)
        klassifier = KNNClassifier(self.best_knn_params_impl[0], self.best_knn_params_impl[1],
                                   self.best_knn_params_impl[2])
        preds = klassifier.fit(self.train_exdog, self.train_ans,
                               self.get_weights(w, self.train_exdog, anomalies1, anomalies2)).predict(x)
        return self.accuracy(y, preds)

    def _objective_lib(self, trial_) -> float:
        k = trial_.suggest_int('k', 1, 500)
        weight_func = trial_.suggest_categorical('weight function', ['uniform', 'distance'])
        algorithm = trial_.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute'])
        p = trial_.suggest_float('minkowski degree', 1.0, 5.0)
        predictions = (KNeighborsClassifier(k, weights=weight_func, algorithm=algorithm, p=p)
                       .fit(self.train_exdog, self.train_ans).predict(self.test_exdog))
        return self.accuracy(self.test_ans, predictions)

    def _objective_impl(self, w_type: WType, trial_) -> float:
        if w_type == WType.FIXED:
            window = FixedWindow(trial_.suggest_float('fixed:h', 0.01, 1))
        else:
            window = RelativeWindow(trial_.suggest_int('relative:k', 1, self.train_ans.__len__() - 1))
        k_type = trial_.suggest_categorical('kernel', [k_type.value for k_type in KType])
        kernel = self.get_kernel_by_enum(k_type)

        metric = trial_.suggest_categorical('metric', [Metric.COSIN, Metric.CHEBYSHEV, Metric.MINKOWSKI])

        klassifier = KNNClassifier(window, kernel, metric)
        preds = klassifier.fit(self.train_exdog, self.train_ans, self.get_fair_weights(self.train_exdog)).predict(
            self.test_exdog)
        return self.accuracy(self.test_ans, preds)

    def _objective_for_fixed(self, trial_: Trial) -> float:
        return self._objective_impl(WType.FIXED, trial_)

    def _objective_for_relative(self, trial_: Trial) -> float:
        return self._objective_impl(WType.RELATIVE, trial_)

    @staticmethod
    def _run_study(objective: Callable[[Trial], float]) -> trial.FrozenTrial:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        return study.best_trial

    @staticmethod
    def get_weights(w: float, x, anomalies1: [int], anomalies2: [int]) -> [float]:
        n_wights = []
        for idx in x.index:
            if idx in anomalies1:
                n_wights.append(w)
            elif idx in anomalies2:
                n_wights.append(0)
            else:
                n_wights.append(1)
        return n_wights

    T = TypeVar('T')

    @staticmethod
    def get_kernel_by_enum(k_type: KType) -> Kernel:
        return KERNEL_MAPPING.get(k_type)()

    @staticmethod
    def get_fair_weights(x: [T]) -> [float]:
        return [1.0 for _ in range(len(x))]

    @staticmethod
    def accuracy(expected: [T], predicted: [T]) -> float:
        if expected.__len__() != predicted.__len__():
            raise ValueError('Dimensions must be equal')

        def indicator(xx):
            return int(xx[0] == xx[1])

        return sum(indicator(e_p) for e_p in zip(expected, predicted)) / predicted.__len__() * 100

    @staticmethod
    def get_anomalies(x: DataFrame, y: [int], params: (Window, Kernel, Metric)) -> ([int], [int], [int]):
        def np_drop(arr, i):
            return arr[i], np.concatenate((arr[0:i], arr[i + 1:]))

        norm, an1, an2 = [], [], []
        for idx in x.index:
            ans, y_drop = np_drop(y, idx)
            pred = (KNNClassifier(params[0], params[1], params[2])
                    .fit(x.drop(index=idx), y_drop, [1.0 for _ in range(len(x))]).predict(x.loc[[idx]])[0])
            if ans == pred:
                norm.append(idx)
            elif abs(ans - pred) == 1:
                an1.append(idx)
            else:
                an2.append(idx)

        return norm, an1, an2
