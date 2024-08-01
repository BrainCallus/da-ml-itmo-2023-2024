from typing import Any

import numpy as np

from model.interactor.abstract_interactor import KlassifierInteractorABC
from model.interactor.maybe import Maybe
from model.klassifier.ridge_regression import RidgeRegression
from model.params.opt_param import *


class RidgeRegressionInteractor(KlassifierInteractorABC):
    def __init__(self, par_alpha: FloatParam = None):
        super().__init__([FloatParam('alpha', 10000.0, 0.0)], [par_alpha])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 1 and len(constructor_param_map) != 2:
            return Maybe('Parameter `alpha`:float required and it is allow to set transform function only')
        if 'alpha' not in constructor_param_map:
            return Maybe('Parameter `alpha`:float required')
        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> RidgeRegression:
        return RidgeRegression(alpha=constructor_param_map['alpha'],
                               pred_transformer=lambda x: np.where(x > 0.0, 1, 0))

    def add_params_to_objective_parameter_map(self, constructor_param_map: Dict[str, Any]) -> Dict[str, Any]:
        # by default identity but could be changed
        return constructor_param_map
