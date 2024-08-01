from typing import Callable, TypeVar, Type, Dict, Tuple, Any

import numpy as np
import optuna
from optuna import Trial
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from model.interactor.abstract_interactor import KlassifierInteractorABC
from logger.logger import Logger

T = TypeVar('T', bound=KlassifierInteractorABC)


class TreeClassifierOptimizer:
    def __init__(self, exdog, endog, test_part: float = 0.25,
                 metric: Callable[[np.ndarray, np.ndarray], float] = f1_score):
        self.train_exdog, self.test_exdog, self.train_ans, self.test_ans = (
            train_test_split(exdog, endog, test_size=test_part, shuffle=False))
        self.metric = metric
        self._current_type_token: Type[T] = type(None)
        self._current_interactor: T = None
        self.modes: Dict[Type[T], Tuple[T, Dict[str, Any]]] = {}
        self.logger = Logger(TreeClassifierOptimizer.__name__)

    @property
    def current_type_token(self):
        return self._current_type_token

    @current_type_token.setter
    def current_type_token(self, value):
        self._current_type_token = value

    @property
    def current_interactor(self):
        return self._current_interactor

    @current_interactor.setter
    def current_interactor(self, value):
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

    def run_study(self, n_trials: int = 1000):
        if self.current_type_token is None:
            raise ValueError('Choose classifier before optimize')
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)
        self.modes[self.current_type_token] = (self.modes[self.current_type_token][0], study.best_trial.params)

        self.logger.info(f'{self.current_type_token} best trial finished with value {study.best_trial.value}; ' +
                         f'best params {study.best_trial.params}')
        return study.best_trial

    def _objective(self, trial_: Trial):
        interactor = self.modes[self._current_type_token][0]
        klassifier = interactor.build_for_objective(trial_)
        klassifier.fit(self.train_exdog, self.train_ans)
        preds = klassifier.predict(self.test_exdog)

        return self.metric(self.test_ans, preds)
