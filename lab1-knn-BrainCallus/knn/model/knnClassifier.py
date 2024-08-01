import numpy as np
from pandas import DataFrame

from model.params.Kernel import Kernel
from model.params.Metric import Metric
from model.params.Window import Window
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(self, window: Window, kernel: Kernel, metric: Metric):
        self.window: Window = window
        self.kernel: Kernel = kernel
        self.metric: Metric = metric
        self.type_cnt: int = 0
        self.weights: [float] = None
        self.train_exdog = None
        self.train_endog: [int] = None

    @property
    def train_exdog(self):
        return self._train_exdog

    @train_exdog.setter
    def train_exdog(self, value: DataFrame):
        self._train_exdog = value

    def fit(self, exdog, endog, weights: [float]):
        if weights is None:
            raise ValueError('Weights must be initialized')

        self.train_exdog = exdog
        self.train_endog = endog
        self.type_cnt = np.unique(self.train_endog).max() + 1
        self.weights = weights
        return self

    def predict(self, target: DataFrame) -> [int]:
        predictions = []
        g_dist, g_classes, g_weights = self._get_nearest_neighbors(target)

        for idd in range(len(target)):
            dist, classes, weights = g_dist[idd], g_classes[idd], g_weights[idd]
            scores = [0 for _ in range(self.type_cnt)]
            for i in range(len(dist) - 1):
                kernel_x = dist[i] / self.window.get_kernel_param_divvisor(dist)
                scores[classes[i]] += self.kernel.eval(kernel_x) * weights[i]
            predictions.append(scores.index(max(scores)))

        return predictions

    def _get_nearest_neighbors(self, target):
        neighbors_cnt = self.window.get_neighbors_cnt(self.train_exdog)
        nn = NearestNeighbors(n_neighbors=neighbors_cnt, metric=self.metric.__name__)
        nn.fit(self.train_exdog)
        dist, ids = nn.kneighbors(target, n_neighbors=neighbors_cnt)
        classes, weights = [], []
        for i in ids:
            classes.append(list(self.train_endog[i]))
            weights.append([self.weights[j] for j in i])
        return dist, classes, weights
