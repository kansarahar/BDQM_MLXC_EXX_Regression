import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """
    Abstract class for the creation of a dataset.

    All derived classes must implement:
    - __init__
    - get_data_train
    - get_data_test
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_data_train(
        self, sample_size: int = -1, shuffle: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_data_test(
        self, sample_size: int = -1, shuffle: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        pass
