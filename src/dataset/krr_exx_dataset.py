import numpy as np
import os
import pickle
import sys
from typing import Literal
from dataset.base_dataset import BaseDataset


class KRRExxDataset(BaseDataset):
    def __init__(
        self,
        file_path: str = "/storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir/subsampling/subsampled_folder_ex",
        system_type: Literal["bulks", "cubic_bulks", "molecules", "all"] = "all",
    ):
        self.file_path = file_path
        self.system_types = (
            [system_type]
            if system_type != "all"
            else ["bulks", "cubic_bulks", "molecules"]
        )

    def get_s(self, dens, sigma):
        grad_rho = np.sqrt(sigma)
        modified_dens = np.array([1e-10 if value < 1e-10 else value for value in dens])
        s = (
            (grad_rho / (modified_dens ** (4 / 3)))
            * 0.5
            * (1 / (3 * np.pi**2)) ** (1 / 3)
        )
        return s

    def get_x(self, dens, sigma):
        grad_rho = np.sqrt(sigma)
        modified_dens = np.array([1e-10 if value < 1e-10 else value for value in dens])
        x = (grad_rho / (modified_dens ** (4 / 3))) * (2 / 9) * (np.pi / 3) ** (1 / 3)
        return x

    def get_fx_chachiyo(self, x):
        fx_num = 3 * (x**2) + np.pi**2 * np.log(x + 1)
        fx_denom = (3 * x + np.pi**2) * np.log(x + 1)
        fx_chachiyo = fx_num / fx_denom
        return fx_chachiyo

    def _get_file_paths(
        self,
        system_type: Literal["bulks", "cubic_bulks", "molecules"],
        type: Literal["training", "validation"] = "training",
    ):
        (
            x_files,
            exLDA_files,
            exx_files,
        ) = [
            list(
                map(
                    lambda s: os.path.join(self.file_path, system_type, d, s),
                    filter(
                        lambda s: s.endswith(".pkl"),
                        os.listdir(
                            os.path.join(
                                self.file_path,
                                system_type,
                                d,
                            )
                        ),
                    ),
                )
            )
            for d in [
                f"X_system_{type}_subsample",
                f"ex_system_{type}_subsample",
                f"y_system_{type}_subsample",
            ]
        ]
        return (
            x_files,
            exLDA_files,
            exx_files,
        )

    def _get_data_from_file_path(self, file_path: str):
        data = []
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return np.array(data)

    def _get_subsampled_data(
        self, type: Literal["training", "validation"], sample_size=-1, shuffle=False
    ):
        x_files = []
        exLDA_files = []
        exx_files = []
        for system_type in self.system_types:
            x, exLDA, exx = self._get_file_paths(system_type, type)
            x_files += x
            exLDA_files += exLDA
            exx_files += exx
        if (
            len(x_files) != len(exLDA_files)
            or len(x_files) != len(exx_files)
            or len(exLDA_files) != len(exx_files)
        ):
            sys.exit(
                f"File number mismatch - X: {len(x_files)}, y: {len(exLDA_files)}, exx: {len(exx_files)}"
            )

        # Figure out how many points to sample per file and the shape of the output
        points_per_file = int(np.ceil(sample_size / len(x_files)))
        x_list = []
        exLDA_list = []
        exx_list = []
        for x_file, exLDA_file, exx_file in zip(x_files, exLDA_files, exx_files):

            # subsample valid points
            x = self._get_data_from_file_path(x_file)
            s = self.get_s(x[:, 0], x[:, 1])
            indices = [i for i in range(len(x)) if s[i] <= 10]
            np.random.shuffle(indices)
            indices = indices[:points_per_file]

            x_list.append(x[indices])
            exLDA_list.append(self._get_data_from_file_path(exLDA_file)[indices])
            exx_list.append(self._get_data_from_file_path(exx_file)[indices])

        x_data = np.vstack(x_list)[:sample_size]
        exLDA_data = np.vstack(exLDA_list)[:sample_size]
        exx_data = np.vstack(exx_list)[:sample_size]

        # randomly shuffle indices again if desired
        if shuffle:
            indices = [i for i in range(len(x_data))]
            np.random.shuffle(indices)
            x_data = x_data[indices]
            exLDA_data = exLDA_data[indices]
            exx_data = exx_data[indices]

        return x_data, exLDA_data, exx_data

    def get_data_train(self, sample_size=-1, shuffle=False):
        x_data, exLDA_data, exx_data = self._get_subsampled_data("training", sample_size, shuffle)
        dens = x_data[:, 0].reshape(-1, 1)
        sq_grad_dens = x_data[:, 1].reshape(-1, 1)

        # todo: figure out the 0.25 factor
        fxx = exx_data / (0.25 * dens * exLDA_data)
        fx_chachiyo = self.get_fx_chachiyo(self.get_x(dens, sq_grad_dens))
        fx_ratio = fxx / fx_chachiyo

        # todo: figure out the meaning of this whole section of logic
        rcut = np.arange(0.5, 3.5, 0.5)
        mcsh_order = np.arange(0, 3, 1)
        index = 2
        for order in mcsh_order:
            for rc in rcut:
                x_data[:, index] = x_data[:, index] * (rc**3)
                index += 1

        return x_data, fx_ratio

    def get_data_test(self, sample_size=-1, shuffle=False):
        return self.get_data_train(sample_size, shuffle)
