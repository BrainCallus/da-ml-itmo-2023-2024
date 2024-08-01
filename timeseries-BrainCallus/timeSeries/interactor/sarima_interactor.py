import pandas as pd
import numpy as np

from sarima.sarima import Sarima


class SarimaInteractor:

    def __init__(self, data, order, seasonal_order, train_part=0.8):
        """
        init interactor

        Parameters
        ----------
        data : dataset
        order : tuple-3 (p, d, q)
        seasonal_order : tuple-4 (P, D, Q, S)
        train_part : part on witch model trained

        """
        self.data = data
        self.date_idx = self.data.index
        self.data_len = len(data)
        self.order = order
        self.seasonal_order = seasonal_order
        self.S = self.seasonal_order[3]
        self.train_len = int(np.ceil(self.data_len * train_part))
        self.test_len = int(self.data_len - self.train_len)
        self.model = self.init_and_fit_model()
        self.train_predicts = None
        self.test_predicts = None
        self.all_predicts = None

    def init_and_fit_model(self):
        model = Sarima(self.data[:-self.test_len], self.order, self.seasonal_order)
        return model.fit()

    def get_train_predicts(self):
        if self.train_predicts is None:
            self.train_predicts = self.model.predict_in_sample(raw=False)
            self.train_predicts_series = pd.Series(self.train_predicts, self.date_idx[self.S + 1: self.train_len])
        return self.train_predicts_series

    def get_test_predicts(self):
        if self.test_predicts is None:
            self.test_predicts = self.model.forecast(self.test_len)
            self.test_predicts_series = pd.Series(self.test_predicts, self.date_idx[self.train_len:])
        return self.test_predicts_series

    def model_aic(self):
        return self.model.get_aic()

    def get_all_predicts(self):
        if self.all_predicts is None:
            self.all_predicts = self.train_predicts_series.combine_first(self.test_predicts_series)
        return self.all_predicts

    def recover_to_origin(self, origin, starter, diff_times=1, logarithm=True):
        undiff = self.inverse_diff(starter.iloc[0], self.all_predicts.iloc[:], diff_times, logarithm)
        self.recovered_preds = origin[:self.S + 2].combine_first(
            pd.Series(undiff, self.date_idx[self.S:]))
        self.recovered_preds_train = self.recovered_preds.iloc[:self.train_len + 1]
        self.recovered_preds_test = self.recovered_preds.iloc[self.train_len + 1:]

    @staticmethod
    def inverse_diff(x_start, x_diff, diff_times, logarithm):
        x = np.r_[x_start, x_diff]
        for i in range(diff_times, x.shape[0]):
            x[i] += x[i - diff_times]
        return np.exp(x) if logarithm else x
