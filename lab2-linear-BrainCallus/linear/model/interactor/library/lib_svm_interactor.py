from typing import Any

from sklearn.svm import LinearSVC

from model.interactor.abstract_interactor import KlassifierInteractorABC
from model.interactor.maybe import Maybe
from model.params.opt_param import *

svm_opt_params = [
    FloatParam('c', 5.0, 0.01),
    FloatParam('tolerance', 0.01, 1e-6),
    IntParam('num_iters', 500, 10)]


class LibSVMInteractor(KlassifierInteractorABC):
    def __init__(self, par_c: FloatParam = None, par_kernel: ObjCategoricalParam = None,
                 par_tol: FloatParam = None, par_iters: IntParam = None):
        super().__init__(svm_opt_params, [par_c, par_kernel, par_tol, par_iters])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 3:
            return Maybe('1 parameter `alpha`:float expected')
        for param_name in map(lambda x: x.name, svm_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')
        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> LinearSVC:
        return LinearSVC(C=constructor_param_map['c'], tol=constructor_param_map['tolerance'],
                         max_iter=constructor_param_map['num_iters'])
