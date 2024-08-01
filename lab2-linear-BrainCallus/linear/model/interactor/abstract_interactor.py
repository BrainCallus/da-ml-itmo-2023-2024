from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar

from optuna import Trial

from model.interactor.maybe import Maybe
from model.klassifier.linear_klassifier import LinearKlassifier
from model.log.logger import Logger
from model.params.opt_param import OptimParam

T = TypeVar('T', bound=LinearKlassifier)


class KlassifierInteractorABC(ABC):
    def __init__(self, default_params: [OptimParam], user_params: [OptimParam]):
        self.optim_params = [pp[1] if pp[0] is None else pp[0] for pp in zip(user_params, default_params)]
        self.logger = Logger(self.__class__.__name__)
        # self.logger.info(f'{self.logger.clazz}')

    def build(self, constructor_param_map: Dict[str, Any]) -> T:
        _verification = self.verify_params(constructor_param_map).map(lambda s:
                                                                      (self.logger.log_error(s),
                                                                       ValueError(s))).get_or_else(None)
        # вот бы do-нотацию сюды..
        return self._internal_build(constructor_param_map)

    def build_for_objective(self, trial: Trial) -> T:
        constructor_params: Dict[str, Any] = {}
        for opt_param in self.optim_params:
            constructor_params[opt_param.name] = opt_param.suggest_trial(trial)
        return self.build(self.add_params_to_objective_parameter_map(constructor_params))

    def add_params_to_objective_parameter_map(self, constructor_param_map: Dict[str, Any]) -> Dict[str, Any]:
        # by default identity but could be changed
        return constructor_param_map

    @abstractmethod
    def _internal_build(self, constructor_param_map: Dict[str, Any]) -> T:
        pass

    @abstractmethod
    def verify_params(self, constructor_param_map: Dict[str, Any]) -> Maybe[str]:
        pass
