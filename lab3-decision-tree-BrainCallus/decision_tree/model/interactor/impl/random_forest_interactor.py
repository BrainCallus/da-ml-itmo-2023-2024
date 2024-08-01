from typing import Any

from model.interactor.abstract_interactor import KlassifierInteractorABC, T
from model.interactor.maybe import Maybe
from model.params.opt_param import *
from model.classifier.rand_forest import *

rand_forest_opt_params = [
    IntParam('max_depth', 50, 1),
    IntParam('min_samples_split', 50, 1),
    IntParam('min_samples_leaf', 50, 1),
    IntParam('bootstrap', 1, 0),
    IntParam('n_estimators', 100, 1),
    ObjCategoricalParam('criterion', [p_type.value for p_type in PartitionType], PARTITION_CRITERION_MAPPING),
    ObjCategoricalParam('feature_limiter', [l_type.value for l_type in LType], FEATURES_LIMITER_MAPPING)
]


class RandomForestImplInteractor(KlassifierInteractorABC):
    def __init__(self, par_max_depth: IntParam = None, par_min_samples_split: IntParam = None,
                 par_min_samples_leaf: IntParam = None, par_boostrap: IntParam = None,
                 par_n_estimators: IntParam = None,
                 par_criterion: ObjCategoricalParam = None, par_feature_limiter: ObjCategoricalParam = None):
        super().__init__(rand_forest_opt_params,
                         [par_max_depth, par_min_samples_split, par_min_samples_leaf, par_boostrap,
                          par_n_estimators, par_criterion, par_feature_limiter])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 7:
            return Maybe('7 parameters expected')

        for param_name in map(lambda x: x.name, rand_forest_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')

        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> T:
        return RandomForestClassifierImpl(max_depth=constructor_param_map['max_depth'],
                                          min_samples_split=constructor_param_map['min_samples_split'],
                                          min_samples_leaf=constructor_param_map['min_samples_leaf'],
                                          bootstrap=False if constructor_param_map['bootstrap'] == 0 else True,
                                          n_estimators=constructor_param_map['n_estimators'],
                                          criterion=constructor_param_map['criterion'],
                                          feature_limiter=constructor_param_map['feature_limiter'])
