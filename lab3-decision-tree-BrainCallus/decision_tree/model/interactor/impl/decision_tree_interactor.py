from typing import Any

from model.interactor.abstract_interactor import KlassifierInteractorABC, T
from model.interactor.maybe import Maybe
from model.params.opt_param import *
from model.classifier.decision_tree import *

decision_tree_opt_params = [
    IntParam('max_depth', 200, 1),
    IntParam('min_samples_split', 50, 2),
    IntParam('min_samples_leaf', 50, 1),
    ObjCategoricalParam('criterion', [p_type.value for p_type in PartitionType], PARTITION_CRITERION_MAPPING),
]


class DecisionTreeImplInteractor(KlassifierInteractorABC):
    def __init__(self, par_max_depth: IntParam = None, par_min_samples_split: IntParam = None,
                 par_min_samples_leaf: IntParam = None, par_criterion: ObjCategoricalParam = None):
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
        return DecisionTreeClassifierImpl(max_depth=constructor_param_map['max_depth'],
                                          min_samples_split=constructor_param_map['min_samples_split'],
                                          min_samples_leaf=constructor_param_map['min_samples_leaf'],
                                          criterion=constructor_param_map['criterion'])
