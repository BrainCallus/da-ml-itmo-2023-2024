from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class PartitionType(Enum):
    ENTROPY = 'entropy'
    GINI = 'gini'
    LOG_LOSS = 'log_loss'


def get_probabilities(y, weights):
    unique = np.arange(np.max(y) + 1)
    return np.array([np.sum(weights[y == c]) for c in unique]) / np.sum(weights)


def calc_entropy(y, weights):
    probabilities = get_probabilities(y, weights)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


def calc_log_loss(y, weights):
    probabilities = get_probabilities(y, weights)
    return -np.sum(probabilities * np.log(probabilities + 1e-9))


class PartitionCriterion(ABC):
    def __init__(self, name: PartitionType):
        self.name = name
        self.n_classes: int = -1

    def eval_gain(self, classes, idx, weights) -> float:
        if self.n_classes <= 0:
            raise ValueError('n_classes must be positive integer')
        return self._internal_eval_gain(classes, idx, weights)

    def eval_impurity(self, ans, weights) -> float:
        if self.n_classes <= 0:
            raise ValueError('n_classes must be positive integer')
        return self._internal_eval_impurity(ans, weights)

    @abstractmethod
    def _internal_eval_gain(self, classes, idx, weights) -> float:
        pass

    @abstractmethod
    def _internal_eval_impurity(self, ans, weights) -> float:
        pass


class EntropyCriterion(PartitionCriterion):
    def __init__(self):
        super().__init__(PartitionType.ENTROPY)

    def _internal_eval_gain(self, classes, idx, weights) -> float:
        total_weight = np.sum(weights)
        weights_left = weights[:idx]
        weights_right = weights[idx:]
        weight_sum_left = np.sum(weights_left)
        weight_sum_right = np.sum(weights_right)
        return (calc_entropy(classes, weights) -
                (weight_sum_left / total_weight) * calc_entropy(classes[:idx], weights_left) -
                (weight_sum_right / total_weight) * calc_entropy(classes[idx:], weights_right))

    def _internal_eval_impurity(self, ans, weights) -> float:
        return calc_entropy(ans, weights)


class GiniCriterion(PartitionCriterion):
    def __init__(self):
        super().__init__(PartitionType.GINI)

    def _internal_eval_gain(self, classes, idx, weights) -> float:
        total_weight = np.sum(weights)
        weights_left = weights[:idx]
        weights_right = weights[idx:]
        weight_sum_left = np.sum(weights_left)
        weight_sum_right = np.sum(weights_right)
        return (self._internal_eval_impurity(classes, weights) -
                (weight_sum_left / total_weight) * self._internal_eval_impurity(classes[:idx], weights_left) -
                (weight_sum_right / total_weight) * self._internal_eval_impurity(classes[idx:], weights_right))

    def _internal_eval_impurity(self, ans, weights) -> float:
        probabilities = get_probabilities(ans, weights)
        return 1 - np.sum(probabilities ** 2)


class LogLossCriterion(PartitionCriterion):
    def __init__(self):
        super().__init__(PartitionType.LOG_LOSS)

    def _internal_eval_gain(self, classes, idx, weights) -> float:
        total_weight = np.sum(weights)
        weights_left = weights[:idx]
        weights_right = weights[idx:]
        return (calc_log_loss(classes, weights) -
                (np.sum(weights_left) / total_weight) * calc_log_loss(classes[:idx], weights_left) -
                (np.sum(weights_right) / total_weight) * calc_log_loss(classes[idx:], weights_right))

    def _internal_eval_impurity(self, ans, weights) -> float:
        return calc_log_loss(ans, weights)


PARTITION_CRITERION_MAPPING = dict(zip([p_type.value for p_type in PartitionType], PartitionCriterion.__subclasses__()))
