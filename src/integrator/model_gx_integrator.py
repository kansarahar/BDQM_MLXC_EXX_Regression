import os
import sys
import pickle
import numpy as np
from typing import Literal

from dataset.exx_dataset import ExxDataset
from integrator.exx_integrator import ExxIntegrator
from regressor.base_regressor import BaseRegressor


class ModelGxIntegrator(ExxIntegrator):
    def __init__(
        self,
        model: BaseRegressor,
        system_type: Literal["bulks", "molecules", "cubic_bulks"],
        descriptor_data_dir_path: str = "/storage/ice-shared/vip-vvi/descriptor_data/",
        exact_exchange_dir_path: str = "/storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir/subsampling/subsampled_folder_ex/",
    ) -> None:
        super().__init__(system_type, descriptor_data_dir_path, exact_exchange_dir_path)
        self.model = model

    def get_total_energy(self, system: str) -> float:
        volume = self._get_subsystem_volume(self.system_type, system)
        x_data, exLDA_data, exx_data = self.dataset.get_data_for_system(
            "all", self.system_type, system
        )
        dens = x_data[:, 0].reshape(-1, 1)
        sq_grad_dens = x_data[:, 1].reshape(-1, 1)
        fx_chachiyo = self.dataset._get_fx_chachiyo(
            self.dataset._get_x(dens, sq_grad_dens)
        )
        gx = self.model.pred(x_data)
        fxx = gx * fx_chachiyo
        exx_pred = 0.25 * dens * exLDA_data * fxx
        return np.sum(exx_pred) * volume
