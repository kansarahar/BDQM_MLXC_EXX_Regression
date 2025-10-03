import numpy as np
from abc import ABC, abstractmethod

class BaseRegressor(ABC):

  @abstractmethod
  def fit(X: np.ndarray, y: np.ndarray):
    pass

  @abstractmethod
  def pred(X: np.ndarray):
    pass