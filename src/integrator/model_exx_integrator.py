import os
import sys
import pickle
import numpy as np

from integrator.base_integrator import BaseIntegrator
from regressor.base_regressor import BaseRegressor
from utils.get_exx_descriptor_data import get_exx_descriptor_data


class ModelExxIntegrator(BaseIntegrator):
    def __init__(
        self,
        model: BaseRegressor,
        descriptor_data_dir_path: str = "/storage/ice-shared/vip-vvi/descriptor_data",
        exact_exchange_dir_path: str = "/storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir",
    ) -> None:
        super().__init__()
        self.model = model
        self.descriptor_data_dir_path = descriptor_data_dir_path

    def _parse_subsystem(self, subsystem: str) -> tuple[str, str]:
        system, subsystem = subsystem.split("/")
        if system not in ["bulks", "molecules", "cubic_bulks"]:
            sys.exit("system must be one of: ['bulks', 'molecules', 'cubic_bulks']")
        if system == "molecules":
            system = "molecules_data"
        subsystem_options = os.listdir(
            os.path.join(self.descriptor_data_dir_path, system)
        )
        if subsystem not in subsystem_options:
            sys.exit(f"subsystem must be one of: {subsystem_options}")
        return system, subsystem

    def _get_subsystem_volume(self, subsystem: str) -> float:
        system, subsystem = self._parse_subsystem(subsystem)
        file_path = os.path.join(
            self.descriptor_data_dir_path, system, subsystem, "sprc-calc.out"
        )
        volume = -1
        with open(file_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith("Volume"):
                volume = float(
                    line.strip()
                    .removeprefix("Volume: ")
                    .removesuffix(" (Bohr^3)")
                    .strip()
                )
                continue
        if not volume > 0:
            sys.exit(f"Volume could not be parsed from file {file_path}")
        return volume

    def _get_subsystem_num_gridpoints(self, subsystem: str) -> float:
        system, subsystem = self._parse_subsystem(subsystem)
        file_path = os.path.join(
            self.descriptor_data_dir_path, system, subsystem, "sprc-calc.out"
        )
        num_points = -1
        with open(file_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith("FD_GRID"):
                fd_grid = line.strip().removeprefix("FD_GRID: ").strip().split(" ")
                fd_grid = [float(item.strip()) for item in fd_grid]
                if len(fd_grid) != 3:
                    sys.exit(f"Invalid FD_GRID size: {fd_grid}")
                num_points = 1
                for i in fd_grid:
                    num_points *= i
                continue
        if not num_points > 0:
            sys.exit(f"The number of points could not be parsed from file {file_path}")

        return num_points

    # def get_energy_density(self, subsystem: str, gridpoint_idx: int) -> float:
    #     """
    #     Get the energy per unit volume at some position
    #     subsystem is of the form <bulks|molecules|cubic_bulks>/<subsystem> e.g. molecules/C_CN_4
    #     """
    #     dV = self._get_subsystem_dV(subsystem)

    def get_total_energy(self, subsystem: str) -> float:
        """
        Get the total energy of the system
        subsystem is of the form <bulks|molecules|cubic_bulks>/<subsystem> e.g. molecules/C_CN_4
        """
        system_type, system_name = self._parse_subsystem(subsystem)
        X, exLDA, exx = get_exx_descriptor_data(system_type, system_name)
        volume = self._get_subsystem_volume(subsystem)
        return np.sum(exx) * volume
