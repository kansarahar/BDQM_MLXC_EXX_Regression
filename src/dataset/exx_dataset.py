import numpy as np
import os
import pickle
import sys
from typing import Literal
from dataset.base_dataset import BaseDataset


class ExxDataset(BaseDataset):
    def _get_subsystem_volume(
        self,
        descriptor_data_dir_path: str,
        system_type: Literal["bulks", "molecules", "cubic_bulks"],
        system: str,
    ) -> float:
        if system_type == "molecules":
            system_type = "molecules_data"
        file_path = os.path.join(
            descriptor_data_dir_path, system_type, system, "sprc-calc.out"
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
        if volume <= 0:
            sys.exit(f"Volume could not be parsed from file {file_path}")
        return volume

    def _get_subsystem_num_gridpoints(
        self,
        descriptor_data_dir_path: str,
        system_type: Literal["bulks", "molecules", "cubic_bulks"],
        system: str,
    ) -> float:
        if system_type == "molecules":
            system_type = "molecules_data"
        file_path = os.path.join(
            descriptor_data_dir_path, system_type, system, "sprc-calc.out"
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

    def _get_s(self, dens, sigma):
        grad_rho = np.sqrt(sigma)
        modified_dens = np.array([1e-10 if value < 1e-10 else value for value in dens])
        s = (
            (grad_rho / (modified_dens ** (4 / 3)))
            * 0.5
            * (1 / (3 * np.pi**2)) ** (1 / 3)
        )
        return s

    def _get_x(self, dens, sigma):
        grad_rho = np.sqrt(sigma)
        modified_dens = np.array([1e-10 if value < 1e-10 else value for value in dens])
        x = (grad_rho / (modified_dens ** (4 / 3))) * (2 / 9) * (np.pi / 3) ** (1 / 3)
        return x

    def _get_fx_chachiyo(self, x):
        fx_num = 3 * (x**2) + np.pi**2 * np.log(x + 1)
        fx_denom = (3 * x + np.pi**2) * np.log(x + 1)
        fx_chachiyo = fx_num / fx_denom
        return fx_chachiyo

    def _get_file_paths(
        self,
        data_type: Literal["training", "validation", "all"],
        system_type: Literal["bulks", "cubic_bulks", "molecules"],
        system: str,
    ):
        """
        Given the data_type (training, validation, all), the system_type (bulks, cubic_bulks, molecules),
        and a valid system, find the path to the file containing that x, exLDA, and exx data
        """
        if (
            system_type not in self.systems.keys()
            or system not in self.systems[system_type]
        ):
            sys.exit(f"System {system_type}/{system} does not exist")
        if data_type != "all":
            data_type = f"{data_type}_subsample"
        x_file_path, exLDA_file_path, exx_file_path = [
            os.path.join(
                self.file_path,
                system_type,
                f"{d}_system_{data_type}",
                f"{system}.pkl",
            )
            for d in ["X", "ex", "y"]
        ]
        return x_file_path, exLDA_file_path, exx_file_path

    def _get_numpy_data_from_file_path(self, file_path: str) -> np.ndarray:
        """
        Load a given file path as a numpy array
        """
        data = []
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return np.array(data)
    
    def _get_data_for_system(
        self,
        data_type: Literal["training", "validation", "all"],
        system_type: Literal["bulks", "cubic_bulks", "molecules"],
        system: str,
        num_points=-1,
        shuffle: bool = False,
    ):
        """
        Given a data_type and a valid system, get the x, exLDA, and exx data for that system.
        If num_points < 0, all points will be considered for sampling. If num_points > 0, only points with s <= 10 are considered.
        """
        x_file_path, exLDA_file_path, exx_file_path = self._get_file_path(
            data_type, system_type, system
        )
        x = self._get_data_from_file_path(x_file_path)
        s = self._get_s(x[:, 0], x[:, 1])
        indices = (
            [i for i in range(len(x)) if s[i] <= 10]  # subsample valid points
            if num_points > 0
            else [i for i in range(len(x))]
        )
        if shuffle:
            np.random.shuffle(indices)
        indices = indices[:num_points]

        x_data = x[indices]
        exLDA_data = self._get_data_from_file_path(exLDA_file_path)[indices]
        exx_data = self._get_data_from_file_path(exx_file_path)[indices]
        return x_data, exLDA_data, exx_data

    def __init__(
        self,
        descriptor_data_dir_path: str = "/storage/ice-shared/vip-vvi/descriptor_data/",
        exact_exchange_dir_path: str = "/storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir/subsampling/subsampled_folder_ex",
        system_type: Literal["bulks", "cubic_bulks", "molecules"] | None = None,
    ):
        self.descriptor_data_dir_path = descriptor_data_dir_path
        self.exact_exchange_dir_path = exact_exchange_dir_path
        self.system_types = (
            [system_type]
            if system_type != None
            else ["bulks", "cubic_bulks", "molecules"]
        )

        # create a list of systems for each system type
        self.systems = {}
        for st in self.system_types:
            exchange_data_file_names = (
                filter(
                    lambda s: s.endswith(".pkl"),
                    os.listdir(
                        os.path.join(
                            self.exact_exchange_dir_path,
                            st,
                            "ex_system_training_subsample",
                        )
                    ),
                ),
            )
            system_names = [s.replace(".pkl", "") for s in exchange_data_file_names]
            self.systems[st] = system_names

        # the inverse of self.systems - create a map from system to its corresponding system type
        self.system_type_from_system = {}
        for st in self.system_types:
            for system in self.systems[st]:
                self.system_type_from_system[system] = st

    def get_available_systems(self) -> list[str]:
        available_systems = []
        for system_type in self.system_types:
            available_systems += self.systems[system_type]
        return available_systems

    def get_dV(self, system: str) -> np.ndarray:
        system_type = self.system_type_from_system[system]
        volume = self._get_subsystem_volume(
            self.descriptor_data_dir_path, system_type, system
        )
        num_points = self._get_subsystem_num_gridpoints(
            self.descriptor_data_dir_path, system_type, system
        )
        return (volume / num_points) * np.ones((num_points, 1))

    def get_descriptors(self, system) -> np.ndarray:
        system_type = self.system_type_from_system[system]
        x_file_path, exLDA_file_path, exx_file_path = self._get_file_path(
            "all", system_type, system
        )
        x_data = self._get_numpy_data_from_file_path(x_file_path)

        # todo: figure out the meaning of this whole section of logic
        rcut = np.arange(0.5, 3.5, 0.5)
        mcsh_order = np.arange(0, 3, 1)
        index = 2
        for order in mcsh_order:
            for rc in rcut:
                x_data[:, index] = x_data[:, index] * (rc**3)
                index += 1
        return x_data

    def get_exchange_energy_density(self, system) -> np.ndarray:
        system_type = self.system_type_from_system[system]
        x_file_path, exLDA_file_path, exx_file_path = self._get_file_path(
            "all", system_type, system
        )
        exx_data = self._get_numpy_data_from_file_path(exx_file_path)
        return exx_data


    def _get_subsampled_data(
        self,
        data_type: Literal["training", "validation", "all"],
        sample_size=-1,
        shuffle=False,
    ):
        # Figure out how many points to sample per file and the shape of the output
        points_per_file = int(np.ceil(sample_size / len(self.get_available_systems())))
        x_list = []
        exLDA_list = []
        exx_list = []
        for system_type in self.system_types:
            for system in self.systems[system_type]:
                x_data, exLDA_data, exx_data = self._get_data_for_system(
                    data_type,
                    system_type,
                    system,
                    num_points=points_per_file,
                    shuffle=shuffle,
                )
                x_list.append(x_data)
                exLDA_list.append(exLDA_data)
                exx_list.append(exx_data)

        x_data = np.vstack(x_list)[:sample_size]
        exLDA_data = np.vstack(exLDA_list)[:sample_size]
        exx_data = np.vstack(exx_list)[:sample_size]

        # todo: figure out the meaning of this whole section of logic
        rcut = np.arange(0.5, 3.5, 0.5)
        mcsh_order = np.arange(0, 3, 1)
        index = 2
        for order in mcsh_order:
            for rc in rcut:
                x_data[:, index] = x_data[:, index] * (rc**3)
                index += 1

        return x_data, exLDA_data, exx_data
    
    def convert_labels_to_exchange_energy_density(self, system, y):
        system_type = self.system_type_from_system[system]
        x_file_path, exLDA_file_path, exx_file_path = self._get_file_path(
            "all", system_type, system
        )
        x_data = self._get_numpy_data_from_file_path(x_file_path)
        exLDA_data = self._get_numpy_data_from_file_path(exLDA_file_path)
        dens = x_data[:, 0].reshape(-1, 1)
        sq_grad_dens = x_data[:, 1].reshape(-1, 1)

        if (len(exLDA_data) != len(y)):
            sys.exit("Size mismatch: y is not the correct size")

        fx_chachiyo = self._get_fx_chachiyo(self._get_x(dens, sq_grad_dens))
        fxx = y / fx_chachiyo
        exx_data = fxx * (0.25 * dens * exLDA_data)

        return exx_data

    def get_data_train(self, sample_size=-1, shuffle=False):
        x_data, exLDA_data, exx_data = self._get_subsampled_data(
            "training", sample_size, shuffle
        )
        dens = x_data[:, 0].reshape(-1, 1)
        sq_grad_dens = x_data[:, 1].reshape(-1, 1)

        # todo: figure out the 0.25 factor
        fxx = exx_data / (0.25 * dens * exLDA_data)
        fx_chachiyo = self._get_fx_chachiyo(self._get_x(dens, sq_grad_dens))
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
