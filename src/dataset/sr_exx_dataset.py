import numpy as np
import os
import pickle
import sys
from typing import Literal
from dataset.base_dataset import BaseDataset


class SRExxDataset(BaseDataset):
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
                f"y_system_{type}_subsample",
                f"ex_system_{type}_subsample",
            ]
        ]
        return (
            x_files,
            exLDA_files,
            exx_files,
        )

    def _get_data_from_file_paths(self, file_paths: list[str]):
        shape = [0, 0]
        for file_path in file_paths:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
                shape[0] += data.shape[0]
                shape[1] = data.shape[1]
        res = np.zeros(shape)
        curr = 0
        for file_path in file_paths:
            with open(file_path, "rb") as file:
                data = pickle.load(file)
                res[curr : curr + data.shape[0], :] = data
                curr += data.shape[0]
        return res

    def _get_data(
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
        exx_data = self._get_data_from_file_paths(exx_files)
        # exLDA_data = self._get_data_from_file_paths(exLDA_files)
        x_data = self._get_data_from_file_paths(x_files)
        indices = [i for i in range(len(exx_data))]
        if sample_size > 0:
            if shuffle:
                np.random.shuffle(indices)
            indices = indices[:sample_size]
            exx_data = exx_data[indices]
            # exLDA_data = exLDA_data[indices]
            x_data = x_data[indices]
        return x_data, exx_data

    def get_data_train(self, sample_size=-1, shuffle=False):
        return self._get_data("training", sample_size, shuffle)

    def get_data_test(self, sample_size=-1, shuffle=False):
        return self._get_data("validation", sample_size, shuffle)
