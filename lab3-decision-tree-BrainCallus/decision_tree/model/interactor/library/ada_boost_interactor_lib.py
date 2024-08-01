from typing import Any

from model.interactor.abstract_interactor import KlassifierInteractorABC, T
from model.interactor.maybe import Maybe
from model.params.opt_param import *
from sklearn.ensemble import AdaBoostClassifier

ada_samme_opt_params = [
    IntParam('n_estimators', 1000, 1)
]


class AdaBoostLibInteractor(KlassifierInteractorABC):
    def __init__(self, base_estimator_getter, par_n_esimators: IntParam = None):
        super().__init__(ada_samme_opt_params, [par_n_esimators])
        self.base_estimator = base_estimator_getter

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 2:
            return Maybe('2 parameters expected')

        for param_name in map(lambda x: x.name, ada_samme_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')

        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> T:
        return AdaBoostClassifier(estimator=constructor_param_map['base_estimator'],
                                  n_estimators=constructor_param_map['n_estimators'], algorithm='SAMME',
                                  random_state=None)

    def add_params_to_objective_parameter_map(self, constructor_param_map: Dict[str, Any]) -> Dict[str, Any]:
        constructor_param_map['base_estimator'] = self.base_estimator
        return constructor_param_map
