from typing import Callable, TypeVar, Type, Dict, Tuple, Any

import numpy as np
import optuna
from optuna import Trial
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from model.interactor.abstract_interactor import KlassifierInteractorABC
from model.log.logger import Logger

T = TypeVar('T', bound=KlassifierInteractorABC)


class LinearClassifierOptimizer:
    def __init__(self, exdog, endog, true_ans=None, test_part: float = 0.25,
                 metric: Callable[[np.ndarray, np.ndarray], float] = f1_score):
        self.train_exdog, self.test_exdog, self.train_ans, self.test_ans, self.train_real_ans, self.test_real_ans = (
            train_test_split(exdog, endog, endog if true_ans is None else true_ans, test_size=test_part, shuffle=False))
        self.metric = metric
        self._current_type_token: Type[T] = type(None)
        self._current_interactor: T = None
        self.modes: Dict[Type[T], Tuple[T, Dict[str, Any]]] = {}
        self.logger = Logger(LinearClassifierOptimizer.__name__)

    @property
    def current_type_token(self):
        return self._current_type_token

    @current_type_token.setter
    def current_type_token(self, value):
        self._current_type_token = value

    @property
    def current_interactor(self):
        return self._current_interactor

    @current_type_token.setter
    def current_type_token(self, value):
        self._current_interactor = value

    def add_mode(self, type_token: Type[T], constructor: T, best_params: Dict[str, Any] = None):
        self.logger.info(f'{type_token.__name__}({type_token}) successfully added(updated)' +
                         ('' if best_params is None else f' with params {best_params}'))
        self.modes[type_token] = (constructor, best_params)

    def switch_mode(self, mode_token: Type[T]):
        if mode_token not in self.modes:
            self.logger.log_error(f'unexpected value {mode_token}. ' +
                                  'Interactor dictionary doesn\'t contain any infromation about provided type')
        else:
            self._current_type_token = mode_token
            self._current_interactor = self.modes[mode_token][0]

    def run_study(self, n_trials: int = 1000, trans: Callable[[np.ndarray], np.ndarray] = None):
        if self.current_type_token is None:
            raise ValueError('Choose klassifier before optimize')
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: self._objective(t, trans), n_trials=n_trials)
        self.modes[self.current_type_token] = (self.modes[self.current_type_token][0], study.best_trial.params)

        self.logger.info(f'{self.current_type_token} best trial finished with value {study.best_trial.value}; ' +
                         f'best params {study.best_trial.params}')
        return study.best_trial

    def _objective(self, trial_: Trial, trans: Callable[[np.ndarray], np.ndarray] = None):
        interactor = self.modes[self._current_type_token][0]
        klassifier = interactor.build_for_objective(trial_)
        klassifier.fit(self.train_exdog, self.train_ans)
        preds = klassifier.predict(self.test_exdog)
        if trans is not None:
            preds = trans(preds)
        return self.metric(self.test_real_ans, preds)
