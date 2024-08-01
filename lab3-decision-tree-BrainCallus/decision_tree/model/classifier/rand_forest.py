from typing import Tuple

from numpy import signedinteger

from logger.logger import Logger
from model.classifier.decision_tree import DecisionTreeClassifierImpl
from model.params.limiter import *
from model.params.partition_crit import *


class RandomForestClassifierImpl:
    def __init__(self, criterion: PartitionCriterion, feature_limiter: FeaturesLimiter,
                 n_estimators: int = 100, max_depth: int = None, min_samples_split=2,
                 min_samples_leaf=1, bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.feature_limiter = feature_limiter
        self.bootstrap = bootstrap
        self.trees = []
        self.n_classes = -1
        self.logger = Logger(RandomForestClassifierImpl.__name__)

    def fit(self, train_exdog, train_ans):
        train_exdog = np.array(train_exdog)
        n_features, max_features = self._reset_before_fit(train_exdog, train_ans)
        mod = int(max(10, self.n_estimators // 10))

        for i in range(self.n_estimators):
            if i % mod == 0:
                self.logger.info(f'Processed {i} / {self.n_estimators} estimators')

            self._process_estimator(train_exdog, train_ans, n_features, max_features)

        self.logger.info(f'Processed {self.n_estimators} / {self.n_estimators} estimators')

    def predict(self, test_exdog):
        test_exdog = np.array(test_exdog)
        tree_predictions = np.array(
            [tree.predict(test_exdog[:, feature_indices]) for tree, feature_indices in self.trees])
        minlength = len(np.unique(tree_predictions))

        return [self._vote(tree_predictions[:, i], minlength) for i in range(tree_predictions.shape[1])]

    def _reset_before_fit(self, train_exdog: np.ndarray, train_ans: np.ndarray) -> Tuple[int, int]:
        self.trees = []
        n_features = train_exdog.shape[1]
        self.n_classes = np.max(train_ans) + 1
        return n_features, self.feature_limiter.limit(n_features)

    def _process_estimator(self, train_exdog, train_ans, n_features, max_features):
        if self.bootstrap:
            x_sample, y_sample = self._bootstrap_sample(train_exdog, train_ans)
        else:
            x_sample, y_sample = train_exdog, train_ans

        features_indices = np.random.choice(n_features, max_features, replace=False)
        x_sample_subset = x_sample[:, features_indices]
        tree = DecisionTreeClassifierImpl(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion
        )
        tree.fit(x_sample_subset, y_sample, n_classes=self.n_classes)
        self.trees.append((tree, features_indices))

    @staticmethod
    def _bootstrap_sample(exdog, ans):
        n_samples = exdog.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return exdog[indices], ans[indices]

    @staticmethod
    def _vote(classes: np.ndarray, min_len: int) -> signedinteger:
        return np.bincount(classes.astype(int), minlength=min_len).argmax()
