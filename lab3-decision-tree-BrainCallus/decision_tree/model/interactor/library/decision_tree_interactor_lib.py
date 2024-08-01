from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from model.interactor.abstract_interactor import KlassifierInteractorABC, T
from model.interactor.maybe import Maybe
from model.params.opt_param import *

decision_tree_opt_params = [
    IntParam('max_depth', np.iinfo(np.int32).max, np.iinfo(np.int32).max),
    IntParam('min_samples_split', 50, 2),
    IntParam('min_samples_leaf', 50, 1),
    StringCategoricalParam('criterion', ['gini', 'entropy', 'log_loss']),
]


class DecisionTreeLibInteractor(KlassifierInteractorABC):
    def __init__(self, par_max_depth: IntParam = None, par_min_samples_split: IntParam = None,
                 par_min_samples_leaf: IntParam = None, par_criterion: StringCategoricalParam = None):
        super().__init__(decision_tree_opt_params,
                         [par_max_depth, par_min_samples_split, par_min_samples_leaf, par_criterion])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 4:
            return Maybe('4 parameters expected')

        for param_name in map(lambda x: x.name, decision_tree_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')

        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> T:
        return DecisionTreeClassifier(max_depth=constructor_param_map['max_depth'],
                                      min_samples_split=constructor_param_map['min_samples_split'],
                                      min_samples_leaf=constructor_param_map['min_samples_leaf'],
                                      criterion=constructor_param_map['criterion'])
