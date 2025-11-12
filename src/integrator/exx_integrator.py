import os
import sys
import pickle
import numpy as np
from typing import Literal

from dataset.exx_dataset import ExxDataset
from integrator.base_integrator import BaseIntegrator
from regressor.base_regressor import BaseRegressor


class ExxIntegrator(BaseIntegrator):
    def __init__(
        self,
        system_type: Literal["bulks", "molecules", "cubic_bulks"],
        descriptor_data_dir_path: str = "/storage/ice-shared/vip-vvi/descriptor_data/",
        exact_exchange_dir_path: str = "/storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir/subsampling/subsampled_folder_ex/",
    ) -> None:
        super().__init__()
        self.dataset = ExxDataset(exact_exchange_dir_path, system_type)
        self.system_type = system_type
        self.descriptor_data_dir_path = descriptor_data_dir_path
        self.exact_exchange_dir_path = exact_exchange_dir_path

    def _get_subsystem_volume(
        self, system_type: Literal["bulks", "molecules", "cubic_bulks"], system: str
    ) -> float:
        if system_type == "molecules":
            system_type = "molecules_data"
        file_path = os.path.join(
            self.descriptor_data_dir_path, system_type, system, "sprc-calc.out"
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

    def _get_subsystem_num_gridpoints(
        self, system_type: Literal["bulks", "molecules", "cubic_bulks"], system: str
    ) -> float:
        if system_type == "molecules":
            system_type = "molecules_data"
        file_path = os.path.join(
            self.descriptor_data_dir_path, system_type, system, "sprc-calc.out"
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

    def get_total_energy(self, system: str) -> float:
        volume = self._get_subsystem_volume(self.system_type, system)
        x, exLDA, exx = self.dataset.get_data_for_system(
            "all", self.system_type, system
        )
        return np.sum(exx) * volume
