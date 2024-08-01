from typing import Any

from sklearn.ensemble import RandomForestClassifier

from model.interactor.abstract_interactor import KlassifierInteractorABC, T
from model.interactor.maybe import Maybe
from model.params.opt_param import *
from model.params.partition_crit import PartitionType

decision_tree_opt_params = [
    IntParam('max_depth', 100, 1),
    IntParam('min_samples_split', 50, 2),
    IntParam('min_samples_leaf', 50, 1),
    IntParam('bootstrap', 1, 0),
    IntParam('n_estimators', 500, 1),
    StringCategoricalParam('criterion', [p_type.value for p_type in PartitionType]),
]


class RandomForestLibInteractor(KlassifierInteractorABC):
    def __init__(self, par_max_depth: IntParam = None, par_min_samples_split: IntParam = None,
                 par_min_samples_leaf: IntParam = None, par_boostrap: IntParam = None,
                 par_n_estimators: IntParam = None, par_criterion: StringCategoricalParam = None):
        super().__init__(decision_tree_opt_params,
                         [par_max_depth, par_min_samples_split, par_min_samples_leaf, par_boostrap,
                          par_n_estimators, par_criterion])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 6:
            return Maybe('6 parameters expected')

        for param_name in map(lambda x: x.name, decision_tree_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')

        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> T:
        return RandomForestClassifier(max_depth=constructor_param_map['max_depth'],
                                      min_samples_split=constructor_param_map['min_samples_split'],
                                      min_samples_leaf=constructor_param_map['min_samples_leaf'],
                                      bootstrap=False if constructor_param_map['bootstrap'] == 0 else True,
                                      n_estimators=constructor_param_map['n_estimators'],
                                      criterion=constructor_param_map['criterion'])
