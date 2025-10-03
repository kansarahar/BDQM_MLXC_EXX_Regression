import numpy as np
from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    """
    Base class for regressors.

    All models must implement:
    - __init__
    - fit
    - pred
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def pred(self, X: np.ndarray) -> np.ndarray:
        pass
