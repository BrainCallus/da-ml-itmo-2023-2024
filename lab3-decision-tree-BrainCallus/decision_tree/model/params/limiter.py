from enum import Enum

import numpy as np


class LType(Enum):
    NONE = 'none'
    SQRT = 'sqrt'
    LOG2 = 'log2'


class FeaturesLimiter:
    def __init__(self, name: LType = LType.NONE):
        self.name = name

    def limit(self, n_features: int) -> int:
        return n_features


class SqrtLimiter(FeaturesLimiter):
    def __init__(self):
        super().__init__(LType.SQRT)

    def limit(self, n_features: int) -> int:
        return int(n_features ** 0.5)


class Log2Limiter(FeaturesLimiter):
    def __init__(self):
        super().__init__(LType.LOG2)

    def limit(self, n_features: int) -> int:
        return int(np.log2(1 if n_features <= 0 else n_features))


FEATURES_LIMITER_MAPPING = dict(zip([l_type.value for l_type in LType],
                                    [FeaturesLimiter] + FeaturesLimiter.__subclasses__()))
