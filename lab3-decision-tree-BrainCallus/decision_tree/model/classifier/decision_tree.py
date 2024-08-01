from typing import Tuple, Optional

from logger.logger import Logger
from model.params.partition_crit import *


class DecisionTreeNode:
    def __init__(self, num_samples: int, predicted_class: int):
        self.num_samples: int = num_samples
        self.predicted_class: int = predicted_class
        self.feature_index: int = 0
        self.threshold: float = 0
        self.left: Optional[DecisionTreeNode] = None
        self.right: Optional[DecisionTreeNode] = None


class DecisionTreeClassifierImpl:
    def __init__(self, criterion: PartitionCriterion, max_depth: int = 1, min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root: DecisionTreeNode | None = None
        self.n_classes: int = -1
        self.n_features: int = -1
        self.logger: Logger = Logger(DecisionTreeClassifierImpl.__name__)

    def fit(self, train_exdog, train_ans, n_classes=None, weights: np.ndarray = None) -> None:
        weights = np.array(weights) if weights is not None else np.ones(len(train_ans))
        if len(weights) != len(train_ans):
            raise ValueError('Expected bijection between weights and answer classes')
        self.n_classes = len(set(train_ans)) if n_classes is None else n_classes
        self.n_features = train_exdog.shape[1]
        self.criterion.n_classes = self.n_classes
        self.root = self._extend_tree(np.array(train_exdog), train_ans, weights)

    def _extend_tree(self, exdog: np.ndarray, ans: np.ndarray, weights: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        num_samples_per_class = [np.sum(weights[ans == i]) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        impurity = self.criterion.eval_impurity(ans, weights)

        node = DecisionTreeNode(num_samples=len(ans), predicted_class=predicted_class)

        if depth < self.max_depth and len(ans) >= self.min_samples_split and impurity > 0:
            idx, thr = self._best_split(exdog, ans, weights, num_samples_per_class)
            if idx is not None:
                indices_left = exdog[:, idx] < thr
                x_left, y_left, w_left = exdog[indices_left], ans[indices_left], weights[indices_left]
                x_right, y_right, w_right = exdog[~indices_left], ans[~indices_left], weights[~indices_left]
                if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                    node.feature_index = idx
                    node.threshold = thr
                    node.left = self._extend_tree(x_left, y_left, w_left, depth + 1)
                    node.right = self._extend_tree(x_right, y_right, w_right, depth + 1)
        return node

    def _best_split(self, exdog: np.ndarray, ans: np.ndarray, weights: np.ndarray,
                    num_samples_per_class: list) -> Tuple[Optional[int], float]:
        best_idx, best_thr = None, None
        best_gain = 0
        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(exdog[:, idx], ans)))
            num_left = [0] * self.n_classes
            num_right = np.copy(num_samples_per_class)

            for i in range(len(ans) - 1):
                num_left[classes[i]] += weights[i]
                num_right[classes[i]] -= weights[i]

                gain = (self.criterion.eval_gain(ans, i + 1, weights)
                        if self.criterion.name == PartitionType.ENTROPY
                        else self.criterion.eval_gain(classes, i + 1, weights))
                if not isinstance(gain, float):
                    self.logger.warn(f'{self.criterion.name} gain: {gain}')
                if thresholds[i + 1] != thresholds[i] and gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_thr = (thresholds[i + 1] + thresholds[i]) / 2

        return best_idx, best_thr

    def predict(self, test_exdog: np.ndarray) -> [int]:
        return [self._internal_predict(inputs) for inputs in np.array(test_exdog)]

    def _internal_predict(self, inputs) -> int:
        node = self.root
        while node.left:
            node = node.left if inputs[node.feature_index] < node.threshold else node.right

        return node.predicted_class
