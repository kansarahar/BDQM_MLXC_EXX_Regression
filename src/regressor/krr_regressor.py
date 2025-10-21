import sys
from sklearn.kernel_ridge import KernelRidge

from regressor.base_regressor import BaseRegressor


class KRRRegressor(BaseRegressor):
    def __init__(self, alpha=0.0001, gamma=0.1, kernel='rbf'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.krr = KernelRidge(kernel=kernel, alpha=self.alpha, gamma=self.gamma)
        self.fitted = False

    def fit(self, X, y):
        self.krr.fit(X, y)
        self.fitted = True

    def pred(self, X):
        if self.fitted is None:
            sys.exit("Model must be fit before pred can be called")
        return self.krr.predict(X)
