from typing import Any

from sklearn.linear_model import RidgeClassifier

from model.interactor.abstract_interactor import KlassifierInteractorABC
from model.interactor.maybe import Maybe
from model.params.opt_param import *


class LibRidgeInteractor(KlassifierInteractorABC):
    def __init__(self, par_alpha: FloatParam = None):
        super().__init__([FloatParam('alpha', 10000.0, 0.0)], [par_alpha])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 1:
            return Maybe('Parameter `alpha`:float required only')
        if 'alpha' not in constructor_param_map:
            return Maybe('Parameter `alpha`:float required')
        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> RidgeClassifier:
        return RidgeClassifier(alpha=constructor_param_map['alpha'])

    def add_params_to_objective_parameter_map(self, constructor_param_map: Dict[str, Any]) -> Dict[str, Any]:
        # by default identity but could be changed
        return constructor_param_map
