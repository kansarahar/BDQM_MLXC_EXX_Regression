import sys
import numpy as np
import pandas as pd
import symantic
from regressor.base_regressor import BaseRegressor


class SyMANTICRegressor(BaseRegressor):
    def __init__(self, operators=None, metrics=[0.06, 0.995]):
        super().__init__()
        self.operators = operators
        self.metrics = metrics
        self.model = symantic.SymanticModel(
            pd.DataFrame(),
            operators=self.operators,
            metrics=self.metrics,
        )
        self.fitted = None

    def fit(self, X, y):
        df = pd.DataFrame(np.column_stack((y, X)))
        df.rename(columns={0: "Target"}, inplace=True)
        df.rename(columns={col: f"n{col}" for col in df.columns}, inplace=True)
        self.model = symantic.SymanticModel(
            df, operators=self.operators, metrics=self.metrics
        )
        self.fitted = self.model.fit()

    def pred(self, X=None):
        if self.fitted is None:
            sys.exit("Model must be fit before pred can be called")
        return self.fitted
