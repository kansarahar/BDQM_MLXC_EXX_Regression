import os
import pickle
import numpy as np
from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    """
    Base class for regressors.

    All models must implement:
    - __init__
    - fit
    - pred
    - save
    - load
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

    def save(self, file_name: str | None = None):
        if file_name == None:
            file_name = self.__class__.__name__ + "_model.pkl"
        dir_name = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_name, "..", "models", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, file_name: str | None = None):
        if file_name == None:
            file_name = self.__class__.__name__ + "_model.pkl"
        dir_name = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_name, "..", "models", file_name)
        with open(file_path, "rb") as file:
            self.__dict__.update(pickle.load(file))
        return self
