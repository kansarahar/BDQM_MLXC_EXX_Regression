# Copied much of this code directly from /storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir/subsampling/system_subsample.py

import numpy as np
import h5py
import pickle
import os
import sys
import time
from typing import Literal


def log_result(log_filename, message):
    f = open(log_filename, "a")
    f.write(message)
    f.close()
    return


def filepath_contains_spin(filepath):
    return "spin" in filepath.lower()


def get_feature_list_hsmp(max_mcsh_order, step_size, max_r):
    """
    Generates lists of filenames for HSMP files based on given MCSH parameters.
    Iterates over spherical harmonics orders and cutoff radii.

    :param max_mcsh_order: Maximum order of spherical harmonics.
    :param step_size: Step size for the radial cutoff.
    :param max_r: Maximum radial cutoff.
    :return: One list of filenames.
    """
    hsmp_filenames = []
    num_features = 0
    for l in range(max_mcsh_order + 1):
        rcut = step_size
        while rcut <= max_r:
            filename = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_0.csv"
            hsmp_filenames.append(filename)
            rcut += step_size
            num_features += 1
    return hsmp_filenames, num_features


def read_hdf5_data(filepath, num_features, hsmp_filenames):
    with h5py.File(filepath, "r") as data:
        # Accessing the groups and datasets
        Nx = data["functional_database/PBE0/metadata/FD_GRID"][0]
        Ny = data["functional_database/PBE0/metadata/FD_GRID"][1]
        Nz = data["functional_database/PBE0/metadata/FD_GRID"][2]
        feature_grp = data["functional_database/PBE0/feature"]


        # Pre-allocating arrays
        grid_points = Nx * Ny * Nz
        feature_arr = np.zeros((grid_points, num_features + 2))

        # Directly slicing data into pre-allocated arrays
        feature_arr[:, 0] = feature_grp["dens"][:]
        feature_arr[:, 1] = feature_grp["sigma.csv"][:]
        for i, feature in enumerate(hsmp_filenames):
            feature_arr[:, i + 2] = feature_grp[feature][:]

        # Extracting exchange energy densities
        exx = feature_grp["exx"][:].reshape(-1, 1)
        ex_lda = -(3 / (4 * np.pi)) * np.power(
            3 * np.pi * np.pi * feature_arr[:, 0], 1 / 3
        )
        ex_lda = ex_lda.reshape(-1, 1)
    return feature_arr, ex_lda, exx


def get_spin_feature_list_hsmp(max_mcsh_order, step_size, max_r):
    """
    Generates lists of filenames for spin up and spin down HSMP files based on given MCSH parameters.
    Iterates over spherical harmonics orders and cutoff radii.

    :param max_mcsh_order: Maximum order of spherical harmonics.
    :param step_size: Step size for the radial cutoff.
    :param max_r: Maximum radial cutoff.
    :return: Two lists of filenames for spin up and spin down.
    """
    hsmp_filenames_spin_up = []
    hsmp_filenames_spin_down = []
    num_features = 0
    for l in range(max_mcsh_order + 1):
        rcut = step_size
        while rcut <= max_r:
            # For spin up (spin_typ = 1)
            filename_up = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_1.csv"
            hsmp_filenames_spin_up.append(filename_up)
            # For spin down (spin_typ = 2)
            filename_down = f"HSMP_l_{l}_rcut_{rcut:.6f}_spin_typ_2.csv"
            hsmp_filenames_spin_down.append(filename_down)
            rcut += step_size
            num_features += 1
    return hsmp_filenames_spin_up, hsmp_filenames_spin_down, num_features


def read_spin_hdf5_data(
    filepath, num_features, hsmp_filenames_spin_up, hsmp_filenames_spin_down
):
    with h5py.File(filepath, "r") as data:
        # Accessing the groups and datasets
        Nx = data["functional_database/PBE0/metadata/FD_GRID"][0]
        Ny = data["functional_database/PBE0/metadata/FD_GRID"][1]
        Nz = data["functional_database/PBE0/metadata/FD_GRID"][2]
        featureUp_grp = data["functional_database/PBE0/feature_spin_up"]
        featureDwn_grp = data["functional_database/PBE0/feature_spin_down"]

        # Pre-allocating arrays
        grid_points = Nx * Ny * Nz
        feature_arr_up = np.zeros((grid_points, num_features + 2))
        feature_arr_down = np.zeros((grid_points, num_features + 2))

        # Directly slicing data into pre-allocated arrays
        feature_arr_up[:, 0] = featureUp_grp["densUp"][:]
        feature_arr_up[:, 1] = featureUp_grp["sigma_up.csv"][:]
        feature_arr_down[:, 0] = featureDwn_grp["densDwn"][:]
        feature_arr_down[:, 1] = featureDwn_grp["sigma_dn.csv"][:]

        for i, featureUp in enumerate(hsmp_filenames_spin_up):
            feature_arr_up[:, i + 2] = featureUp_grp[featureUp][:]
        for i, featureDwn in enumerate(hsmp_filenames_spin_down):
            feature_arr_down[:, i + 2] = featureDwn_grp[featureDwn][:]

        # Extracting exchange energy densities
        exxUp = featureUp_grp["exxUp"][:].reshape(-1, 1)
        ex_ldaUp = -(3 / (4 * np.pi)) * np.power(
            6 * np.pi * np.pi * feature_arr_up[:, 0], 1 / 3
        )
        ex_ldaUp = ex_ldaUp.reshape(-1, 1)
        exxDwn = featureDwn_grp["exxDwn"][:].reshape(-1, 1)
        ex_ldaDwn = -(3 / (4 * np.pi)) * np.power(
            6 * np.pi * np.pi * feature_arr_down[:, 0], 1 / 3
        )
        ex_ldaDwn = ex_ldaDwn.reshape(-1, 1)

        feature_arr = np.vstack((feature_arr_up, feature_arr_down))
        exx = np.vstack((exxUp, exxDwn))
        ex_lda = np.vstack((ex_ldaUp, ex_ldaDwn))
    return feature_arr, ex_lda, exx


def get_exx_descriptor_data(
    system_type: Literal["bulks", "molecules", "cubic_bulks"],
    system_name: str,
    raw_filepath="/storage/ice-shared/vip-vvi/exact_exchange_work/test_2_dir/data_preparation/hdf5_format_data",
):
    base_path = os.path.join(raw_filepath, system_type)
    hdf5_filepaths = os.listdir(base_path)  # add path to the hdf5 files
    path = [p for p in hdf5_filepaths if p.split("_HSMP")[0] == system_name][0]

    count = 0
    mcsh_max_order = 2
    mcsh_step_size = 0.5
    mcsh_max_r = 3.0

    hsmp_filenames, num_features = get_feature_list_hsmp(
        mcsh_max_order, mcsh_step_size, mcsh_max_r
    )
    hsmp_filenames_spin_up, hsmp_filenames_spin_down, num_spin_features = (
        get_spin_feature_list_hsmp(mcsh_max_order, mcsh_step_size, mcsh_max_r)
    )

    if filepath_contains_spin(path):
        (X, ex_lda, exx) = read_spin_hdf5_data(
            os.path.join(base_path, path),
            num_spin_features,
            hsmp_filenames_spin_up,
            hsmp_filenames_spin_down,
        )
    else:
        (X, ex_lda, exx) = read_hdf5_data(
            os.path.join(base_path, path), num_features, hsmp_filenames
        )

    return X, ex_lda, exx

