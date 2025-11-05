import os
import pickle
import numpy as np
from abc import ABC, abstractmethod


class BaseIntegrator(ABC):
    """
    Base class for integrators.

    All derived classes must implement:
    - __init__
    - get_energy_density
    - get_total_energy
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    # @abstractmethod
    # def get_energy_density(self, subsystem: str, gridpoint_idx: int) -> float:
    #     pass

    @abstractmethod
    def get_total_energy(self, subsystem: str) -> float:
        pass

