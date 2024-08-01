from typing import Any

from sklearn.linear_model import SGDClassifier

from model.interactor.abstract_interactor import KlassifierInteractorABC
from model.interactor.maybe import Maybe
from model.log.logger import Logger
from model.params.opt_param import *
from model.params.loss_function import *

gradient_descent_opt_params = [
    FloatParam('alpha', 1000.0, 0.0),
    FloatParam('beta', 1.0, 0.0),
    FloatParam('learning_rate', 0.1, 1e-5),
    IntParam('num_iters', 1000, 10),
    StringCategoricalParam('loss_function', [f_type.value for f_type in FunType])
]


#  todo : parameter for loss function
#    LossFunc(loss_f, grad_loss, name)


class LibGDInteractor(KlassifierInteractorABC):
    def __init__(self, par_alpha: FloatParam = None, par_beta: FloatParam = None,
                 par_learning_rate: FloatParam = None, par_num_iters: IntParam = None,
                 par_loss_func: StringCategoricalParam = None):

        super().__init__(gradient_descent_opt_params,
                         [par_alpha, par_beta, par_learning_rate, par_num_iters, par_loss_func])

    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        if len(constructor_param_map) != 5:
            return Maybe('5 parameters expected')
        for param_name in map(lambda x: x.name, gradient_descent_opt_params):
            if param_name not in constructor_param_map:
                return Maybe(f'Parameter \'{param_name}\' required')
        return Maybe(None)

    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> SGDClassifier:
        return SGDClassifier(alpha=constructor_param_map['alpha'], l1_ratio=constructor_param_map['beta'],
                             learning_rate='constant', eta0=constructor_param_map['learning_rate'],
                             max_iter=constructor_param_map['num_iters'],
                             loss=constructor_param_map['loss_function']
                             )
