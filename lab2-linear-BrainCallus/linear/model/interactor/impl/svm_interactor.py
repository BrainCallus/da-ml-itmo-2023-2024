from typing import Any

from model.interactor.abstract_interactor import KlassifierInteractorABC
from model.interactor.maybe import Maybe
from model.klassifier.svm import SVM
from model.log.logger import Logger
from model.params.opt_param import *
from model.params.kernel import *

svm_opt_params = [
    FloatParam('c', 5.0, 0.01),
    ObjCategoricalParam('kernel', [k_type.value for k_type in KType], KERNEL_MAPPING),
    FloatParam('tolerance', 0.01, 1e-6),
    IntParam('num_iters', 500, 10)]


class SVMInteractor(KlassifierInteractorABC):
    def __init__(self, par_c: FloatParam = None, par_kernel: ObjCategoricalParam = None,
                 par_tol: FloatParam = None, par_iters: IntParam = None):
        super().__init__(svm_opt_params, [par_c, par_kernel, par_tol, par_iters])

    def build_for_objective(self, trial_: Trial):
        constructor_params: Dict[str, Any] = {}
        for opt_param in self.optim_params:
            if opt_param.name == 'kernel':
                k_type = trial_.suggest_categorical(opt_param.name, opt_param.values)
                if k_type == 'Polynomial':
                    deg = trial_.suggest_int('deg', low=1, high=15)
                    c_par = PolynomialKernel(degree=deg)
                elif k_type == 'Gaussian':
                    gamma = trial_.suggest_float('gamma', low=0.001, high=0.99)
                    c_par = GaussianKernel(gamma)
                else:
                    c_par = opt_param.mapping.get(k_type)()
            else:
                c_par = opt_param.suggest_trial(trial_)
            constructor_params[opt_param.name] = c_par
        return self.build(self.add_params_to_objective_parameter_map(constructor_params))

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 4 and len(constructor_param_map) != 5:
            return Maybe('1 parameter `alpha`:float expected')
        for param_name in map(lambda x: x.name, svm_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')
        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> SVM:
        return SVM(c=constructor_param_map['c'], kernel=constructor_param_map['kernel'],
                   tolerance=constructor_param_map['tolerance'], num_iters=constructor_param_map['num_iters'],
                   pred_transformer=lambda x: np.where(x > 0.0, 1, 0))
