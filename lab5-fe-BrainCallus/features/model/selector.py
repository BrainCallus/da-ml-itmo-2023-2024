import time
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from logger.logger import Logger


def evaluate_model(model, train_x, test_x, ans_train, ans_test, metric: Callable[[[float], [float]], float]):
    model.fit(train_x, ans_train)
    y_pred = model.predict(test_x)
    return metric(ans_test, y_pred)


class SelectorAbc(ABC):
    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

    @abstractmethod
    def select(self, train_exdog, test_exdog, train_ans, test_ans, n_features: int) -> [int]:
        pass


class SVM_RFE(SelectorAbc):
    def select(self, train_exdog, test_exdog, train_ans, test_ans, n_features: int) -> [int]:
        _n_samples, n_f = train_exdog.shape
        cur_selected_features = np.arange(n_f)

        i = 0
        while len(cur_selected_features) > n_features:
            svm_model = SVC(kernel='linear', tol=1e-4, C=2.0)
            svm_model.fit(train_exdog[:, cur_selected_features], train_ans)

            weights = np.abs(svm_model.coef_[0].A[0])

            cur_step = min(len(weights), min(len(cur_selected_features) - n_features, 15))
            min_weight_indices = np.argsort(weights)[:cur_step]

            cur_selected_features = np.delete(cur_selected_features, min_weight_indices)

            if i % 50 == 0:
                self.logger.info(f'Number of features selected: {len(cur_selected_features)} / {n_features}')
            i += 1

        return cur_selected_features


class WrapperForwardSelector(SelectorAbc):
    def select(self, train_exdog, test_exdog, train_ans, test_ans, n_features: int) -> [int]:
        logistic = LogisticRegression(solver='liblinear')
        selected = []
        unselected_features = np.arange(train_exdog.shape[1])

        for i in range(n_features):
            scores = []

            start_exec = time.time()
            for j in range(len(unselected_features)):
                new_feature = unselected_features[j]
                subset_features = np.copy(selected)
                subset_features = np.append(subset_features, new_feature)
                score = evaluate_model(logistic, train_exdog[:, subset_features],
                                       test_exdog[:, subset_features],
                                       train_ans, test_ans, accuracy_score)
                scores.append((score, j, new_feature))

            mx, arg_mx, selected_feature = self._get_arg_max(scores)
            unselected_features = np.delete(unselected_features, arg_mx)
            selected.append(selected_feature)

            self.logger.info(f'On epoch {i}/{n_features}\t elapsed {time.time() - start_exec:.3f} s\t' +
                             f'Added feature {selected_feature} with scores {mx:.5f}')

        return selected

    @staticmethod
    def _get_arg_max(scores_: [Tuple[float, int, int]]) -> Tuple[float, int, int]:
        max_v = -100
        max_idx = -1
        sf = -1
        for sc in scores_:
            if sc[0] > max_v or max_idx == -1:
                max_v = sc[0]
                max_idx = sc[1]
                sf = sc[2]
        return max_v, max_idx, sf


class FilterSelector(SelectorAbc):
    def select(self, train_exdog, test_exdog, train_ans, test_ans, n_features: int) -> [int]:
        chi2_selector = SelectKBest(chi2, k=n_features)
        chi2_selector.fit(train_exdog, train_ans)
        return chi2_selector.get_support(indices=True)


class SelectFromModelRF(SelectorAbc):
    def select(self, train_exdog, test_exdog, train_ans, test_ans, n_features: int) -> [int]:
        rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        rand_forest.fit(train_exdog, train_ans)
        selector = SelectFromModel(rand_forest, threshold=-np.inf, max_features=n_features, prefit=True)
        selector.transform(train_exdog)
        selector.transform(test_exdog)

        return selector.get_support(indices=True)


class VarianceThresholdSelector(SelectorAbc):
    def __init__(self, p: float = 0.9):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError('Illegal probability value; expected in [0.0, 1.0]')
        self.p = p

    def select(self, train_exdog, test_exg, train_ans, test_ans, n_features: int) -> [int]:
        vt_selector = VarianceThreshold(threshold=(self.calc_var()))
        vt_selector.fit_transform(train_exdog)
        return vt_selector.get_support(indices=True)[:n_features]

    def calc_var(self):
        return self.p * (1 - self.p)


class MutualInfoSelector(SelectorAbc):
    def select(self, train_exdog, test_exdog, train_ans, test_ans, n_features: int) -> [int]:
        mi_selector = SelectKBest(mutual_info_classif, k=n_features)
        mi_selector.fit(train_exdog, train_ans)
        return mi_selector.get_support(indices=True)
