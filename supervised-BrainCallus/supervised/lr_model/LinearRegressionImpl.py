import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegressionImpl:
    def __init__(self, exdog, endog, test_part=0.2, shuffle=True):
        self.exdog = exdog
        self.endog = endog
        self.data_len = self.endog.__len__()
        self.test_part = test_part
        self.train_exdog, self.test_exdog, self.train_ans, self.test_ans = train_test_split(self.exdog, self.endog,
                                                                                            test_size=test_part,
                                                                                            random_state=42,
                                                                                            shuffle=shuffle)
        self.train_len = self.train_ans.__len__()
        self.test_len = self.test_ans.__len__()
        self.coefs = None
        self.free_member = None
        self.residuals_train = None
        self.test_endog = None
        self.residuals_test = None

    def fit(self):
        self.regressor = LinearRegression(fit_intercept=True)
        self.regressor.fit(self.train_exdog, self.train_ans)
        self.coefs = self.regressor.coef_
        self.free_member = self.regressor.intercept_
        self.residuals_train = self.train_ans - (self.free_member + np.dot(self.train_exdog, self.coefs))

    def predict(self):
        self.test_endog = self.free_member + np.dot(self.test_exdog, self.coefs)
        self.residuals_test = self.test_ans - self.test_endog

    def get_coefs_as_map(self):
        to_return = {}
        for name, coef in zip(self.exdog.columns.values, self.coefs):
            to_return[name] = coef
        return to_return
