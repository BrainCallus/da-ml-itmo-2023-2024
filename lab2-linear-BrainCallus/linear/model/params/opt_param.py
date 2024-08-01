from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Dict, Type

from optuna import Trial


class OptimParam[R](ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def suggest_trial(self, trial_: Trial) -> R:
        pass


class FloatParam(OptimParam[float]):
    def __init__(self, name: str, up_bnd: float, lw_bnd: float):
        super().__init__(name)
        self.up_bnd = up_bnd
        self.lw_bnd = lw_bnd

    def suggest_trial(self, trial_: Trial) -> float:
        return trial_.suggest_float(self.name, self.lw_bnd, self.up_bnd)


class IntParam(OptimParam[int]):
    def __init__(self, name: str, up_bnd: int, lw_bnd: int):
        super().__init__(name)
        self.up_bnd: int = up_bnd
        self.lw_bnd: int = lw_bnd

    def suggest_trial(self, trial_: Trial) -> int:
        return trial_.suggest_int(self.name, self.lw_bnd, self.up_bnd)


E = TypeVar('E', bound=Enum)


class BaseCategoricalParam[T](OptimParam[T], ABC):
    def __init__(self, name: str, values: [E]):
        super().__init__(name)
        self.values = values

    @abstractmethod
    def suggest_trial(self, trial_: Trial) -> T:
        pass


class StringCategoricalParam(BaseCategoricalParam[str]):
    def __init__(self, name: str, values: [E]):
        super().__init__(name, values)

    def suggest_trial(self, trial_: Trial) -> str:
        return trial_.suggest_categorical(self.name, self.values)


class ObjCategoricalParam[T](BaseCategoricalParam[T]):
    def __init__(self, name: str, values: [E], mapping: Dict[str, Type[T]]):
        super().__init__(name, values)
        self.mapping = mapping

    def suggest_trial(self, trial_: Trial) -> T:
        return self.mapping.get(trial_.suggest_categorical(self.name, self.values))()
