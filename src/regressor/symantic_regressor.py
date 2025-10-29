import sys
import numpy as np
import pandas as pd
import symantic

from regressor.base_regressor import BaseRegressor


class SyMANTICRegressor(BaseRegressor):
    def __init__(self, scaler = None, operators=None, metrics=[0.06, 0.995]):
        super().__init__()
        self.operators = operators
        self.metrics = metrics
        self.model = symantic.SymanticModel(
            pd.DataFrame(),
            operators=self.operators,
            metrics=self.metrics,
        )
        self.scaler = scaler
        self.fitted = False
        self.fitted_results = None

    def fit(self, X, y):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        df = pd.DataFrame(np.column_stack((y, X)))
        df.rename(columns={0: "Target"}, inplace=True)
        df.rename(columns={col: f"n{col}" for col in df.columns}, inplace=True)
        self.model = symantic.SymanticModel(
            df, operators=self.operators, metrics=self.metrics
        )
        self.fitted_results = self.model.fit()
        self.fitted = True

    def pred(self, X=None):
        if self.fitted is False:
            sys.exit("Model must be fit before pred can be called")
        if self.scaler:
            X = self.scaler.transform(X)
