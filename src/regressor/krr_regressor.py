import sys
import pickle
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from regressor.base_regressor import BaseRegressor


class KRRRegressor(BaseRegressor):
    def __init__(self, alpha=0.0001, gamma=0.1, kernel="rbf"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.krr = KernelRidge(kernel=kernel, alpha=self.alpha, gamma=self.gamma)
        self.fitted = False
        self.standard_scaler = StandardScaler()

    def fit(self, X, y):
        transformed_X = self.standard_scaler.fit_transform(X)
        transformed_X = X
        self.krr.fit(transformed_X, y)
        self.fitted = True

    def pred(self, X):
        if self.fitted is None:
            sys.exit("Model must be fit before pred can be called")
        transformed_X = self.standard_scaler.transform(X)
        transformed_X = X
        return self.krr.predict(transformed_X)
