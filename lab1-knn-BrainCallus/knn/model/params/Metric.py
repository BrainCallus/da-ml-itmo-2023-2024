from enum import Enum
import scipy


class Metric(Enum):
    COSIN = scipy.spatial.distance.cosine
    CHEBYSHEV = scipy.spatial.distance.chebyshev
    MINKOWSKI = scipy.spatial.distance.minkowski
