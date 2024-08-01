import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from logger.logger import Logger


class DataHandler:
    def __init__(self, path, target_col):
        self.logger = Logger(DataHandler.__name__)
        self.data: DataFrame = pd.read_csv(path)
        self.target_col: str = target_col
        self.working_copy: DataFrame = self.data.copy()
        self.encoder: LabelEncoder = LabelEncoder()

    def make_ex_endog(self):
        endog = self.working_copy[self.target_col]
        self.drop_cols([self.target_col])
        exdog = pd.get_dummies(self.working_copy, columns=self._get_object_col_names(), dtype=float)
        self.working_copy[self.target_col] = endog
        return exdog, self._linear_encode(endog) if self.is_object_col(endog) else endog

    def drop_cols(self, cols: [str]):
        if not all(col in self.working_copy.keys() for col in cols):
            self.logger.log_error(f'Unexpected names. Working data has only {self.working_copy.keys()}')
        self.working_copy = self.working_copy.drop(cols, axis=1)

    def make_correlation(self):
        cpy = self.working_copy.copy()
        for col_name in self._get_object_col_names():
            cpy[col_name] = self._linear_encode(cpy[col_name])

        self.show_correlation(cpy)

    def logarithm(self, col_name: str):
        if not (col_name in self.working_copy.keys()):
            self.logger.log_error(f'Column with name \'{col_name}\' not found')
            return
        if self.is_object_col(self.working_copy[col_name]):
            self.logger.log_error('Illegal column, numerical type expected')
            return
        self.working_copy[col_name] = np.log(self.working_copy[col_name])

    def _linear_encode(self, col: Series):
        self.encoder.fit(col)
        return self.encoder.transform(col)

    def _get_object_col_names(self) -> [str]:
        return list(filter(lambda key: self.is_object_col(self.working_copy[key]), self.working_copy.keys()))

    @staticmethod
    def show_correlation(df: DataFrame):
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(df.corr(), annot=True, fmt=".2f")
        ii, kk = ax.get_ylim()
        ax.set_ylim(ii + 0.5, kk - 0.5)
        plt.show()

    @staticmethod
    def is_object_col(col: Series) -> bool:
        return col.dtype.name == 'object'
