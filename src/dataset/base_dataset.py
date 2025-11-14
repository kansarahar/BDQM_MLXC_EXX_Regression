import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """
    Abstract class for the creation of a dataset.

    All derived classes must implement the following functions:
    - __init__
    - get_available_systems
    - get_atoms_in_system
    - get_dV
    - get_descriptors
    - get_exchange_energy_density
    - convert_labels_to_energy_density
    - get_data_train
    - get_data_test

    get_data_train and get_data_test are the only two functions that apply to more
    than one system.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_available_systems(self) -> list[str]:
        """
        Return a list of all available systems
        """
        pass

    @abstractmethod
    def get_atoms_in_system(self, system: str) -> dict[str, int]:
        """
        Return a dict containing the number of each atom in the system
        """
        pass

    @abstractmethod
    def get_dV(self, system: str) -> np.ndarray:
        """
        Get the volume element associated with each point in the system (in Bohrs)
        """
        pass

    @abstractmethod
    def get_descriptors(self, system: str) -> np.ndarray:
        """
        Get the descriptors (X) associated with each point in the system.
        The returned value should have shape (N, d), where:
        - N is the number of points in the system
        - d is the number of descriptors per point
        """
        pass

    @abstractmethod
    def get_exchange_energy_density(self, system: str) -> np.ndarray:
        """
        Get the exchange energy density (exx) associated with each point in the
        system (in Hartrees). The returned value should have shape (N, 1), where:
        - N is the number of points in the system
        """

    @abstractmethod
    def convert_labels_to_exchange_energy_density(
        system: str, y: np.ndarray
    ) -> np.ndarray:
        """
        For every point in the given system, convert the train/test labels y into
        exx. If the labels are already equal to exx, simply return y.
        """
        return

    @abstractmethod
    def get_data_train(
        self, sample_size: int = -1, shuffle: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a subset of the data (X, y) that can be used to train a regression model.
        The return values will be X with shape (N, d) and y with shape (N, 1), where:
        - N is the size of the subset of points to be used for training
        - d is the number of descriptors per point

        Note that y does not necessarily have to be exx as the model can learn a
        different function that can be transformed into exx using
        self.convert_labels_to_energy_density.
        """
        pass

    @abstractmethod
    def get_data_test(
        self, sample_size: int = -1, shuffle: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a subset of the data (X, y) that can be used to test a regression model.
        The return values will be X with shape (N, d) and y with shape (N, 1), where:
        - N is the size of the subset of points to be used for training
        - d is the number of descriptors per point

        Note that y does not necessarily have to be exx as the model can learn a
        different function that can be transformed into exx using
        self.convert_labels_to_energy_density.
        """
        pass
