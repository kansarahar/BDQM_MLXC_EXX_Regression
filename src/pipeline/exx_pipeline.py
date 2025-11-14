import numpy as np
from sympy import symbols, Eq, solve, collect, expand, linear_eq_to_matrix
import sys
from dataset.base_dataset import BaseDataset
from regressor.base_regressor import BaseRegressor


class ExxPipeline:
    """
    Class with helpful methods for training / validation models to learn exx
    """

    def __init__(
        self, dataset: BaseDataset, regressor: BaseRegressor | None = None
    ) -> None:
        self.dataset = dataset
        self.regressor = regressor

    def set_regressor(self, regressor: BaseRegressor) -> None:
        self.regressor = regressor

    def get_available_systems(self) -> list[str]:
        return self.dataset.get_available_systems()

    def get_true_total_exchange_energy(self, system: str) -> float:
        exx = self.dataset.get_exchange_energy_density(system)
        dV = self.dataset.get_dV(system)
        return float(np.sum(exx * dV))

    def train_regressor(
        self, sample_size=-1, shuffle_data=True, model_name: str | None = None
    ) -> None:
        if self.regressor == None:
            sys.exit("No regressor associated with this pipeline")
        print("Loading training data...")
        X_train, y_train = self.dataset.get_data_train(
            sample_size, shuffle=shuffle_data
        )
        print("Loading testing data...")
        X_test, y_test = self.dataset.get_data_test(sample_size, shuffle=shuffle_data)

        print("Fitting model to data")
        self.regressor.fit(X_train, y_train)
        print(f"Saving model as {model_name}")
        self.regressor.save(model_name)

        train_pred = self.regressor.pred(X_train)
        test_pred = self.regressor.pred(X_test)
        print(f"Training MSE: {np.mean((train_pred - y_train)**2)}")
        print(f"Testing MSE: {np.mean((test_pred - y_test)**2)}")

    def get_predicted_total_exchange_energy(
        self, system: str, model_name: str | None = None
    ) -> float:
        if self.regressor == None:
            sys.exit("No regressor associated with this pipeline")
        # load trained model
        self.regressor.load(model_name)

        X = self.dataset.get_descriptors(system)
        y = self.regressor.pred(X)

        exx = self.dataset.convert_labels_to_exchange_energy_density(system, y)
        dV = self.dataset.get_dV(system)
        return float(np.sum(exx * dV))

    def get_reaction_coefficients(
        self, reactants: list[str], products: list[str]
    ) -> tuple[list[int], list[int]]:

        reactant_atoms = [
            self.dataset.get_atoms_in_system(reactant) for reactant in reactants
        ]
        product_atoms = [
            self.dataset.get_atoms_in_system(product) for product in products
        ]

        # ensure that the same atoms appear on the reactant side as the product side
        reactant_atoms_set = set()
        product_atoms_set = set()

        for mol in reactant_atoms:
            for atom in mol.keys():
                reactant_atoms_set.add(atom)

        for mol in product_atoms:
            for atom in mol.keys():
                product_atoms_set.add(atom)

        if reactant_atoms_set != product_atoms_set:
            sys.exit(
                "Invalid reaction: missing atoms from either products or reactants"
            )

        # generate symbols for each atom
        atom_list = list(reactant_atoms_set)
        atom_symbols_list = symbols(" ".join(atom_list), seq=True)

        # generate symbols for the coefficients on each molecule
        reactant_coeff_symbols_list = symbols(
            " ".join([f"r{i}" for i in range(len(reactants))]), seq=True
        )
        product_coeff_symbols_list = symbols(
            " ".join([f"p{i}" for i in range(len(products))]), seq=True
        )
        atom_symbols_map = {
            atom_list[i]: sym for i, sym in enumerate(atom_symbols_list)
        }

        # generate a symbolic expression for the total number of atoms on the reactant side
        expr = 0
        for mol_idx, mol in enumerate(reactant_atoms):
            for atom, count in mol.items():
                expr += (count * atom_symbols_map[atom]) * reactant_coeff_symbols_list[
                    mol_idx
                ]
        # subtract from that a symbolic expression for the total number of atoms on the product side
        for mol_idx, mol in enumerate(product_atoms):
            for atom, count in mol.items():
                expr -= (count * atom_symbols_map[atom]) * product_coeff_symbols_list[
                    mol_idx
                ]
        total = expand(expr)

        # recognize that atoms are conserved, so the total number of each atom in the expression should be zero
        collected = collect(total, atom_symbols_list, evaluate=False)
        equations = [Eq(collected.get(v, 0), 0) for v in atom_symbols_list]

        # express the above equations as a linear equation and solve
        coeff_symbols_list = reactant_coeff_symbols_list + product_coeff_symbols_list
        A, b = linear_eq_to_matrix(list(collected.values()), coeff_symbols_list)
        A = A.rref()[0]
        A, b = np.array(A, dtype=float), np.array(b, dtype=float)
        A, b = (
            A[~np.all(A == 0, axis=1)],
            b[~np.all(A == 0, axis=1)],
        )  # remove rows that are all zeros

        # system must be underdetermined to be a valid chemical equation
        if A.shape[0] >= A.shape[1]:
            sys.exit("Chemical equation cannot be balanced")

        # set the final product coeff to always be 1
        # I am choosing not to handle the case where the system might still be underdetermined after this as it is probably irrelevant
        A = np.append(A, np.zeros((1, A.shape[1])), axis=0)
        A[-1, -1] = 1
        b = np.append(b, [[1]], axis=0)

        coeffs = np.linalg.solve(A, b).T[0]
        reactant_coeffs = coeffs[: len(reactants)]
        product_coeffs = coeffs[len(reactants) :]

        # print reaction details
        for i, reactant in enumerate(reactants):
            print(f"{reactant}:", reactant_atoms[i])
        for i, product in enumerate(products):
            print(f"{product}:", product_atoms[i])
        reactant_strings = [
            f"{reactant_coeffs[i]}({reactants[i]})" for i in range(len(reactants))
        ]
        product_strings = [
            f"{product_coeffs[i]}({products[i]})" for i in range(len(products))
        ]
        reaction_string = (
            f"{' + '.join(reactant_strings)} => {' + '.join(product_strings)}"
        )
        print("RXN:", reaction_string)

        return reactant_coeffs, product_coeffs

    def get_true_reaction_exchange_energy(
        self, reactants: list[str], products: list[str]
    ):
        reactant_coeffs, product_coeffs = self.get_reaction_coefficients(
            reactants, products
        )
        reactant_energies = [
            self.get_true_total_exchange_energy(system)
            for i, system in enumerate(reactants)
        ]
        product_energies = [
            self.get_true_total_exchange_energy(system)
            for i, system in enumerate(products)
        ]
        return sum(product_energies) - sum(reactant_energies)

    def get_predicted_reaction_exchange_energy(
        self, reactants: list[str], products: list[str], model_name: str | None = None
    ):
        reactant_coeffs, product_coeffs = self.get_reaction_coefficients(
            reactants, products
        )
        reactant_energies = [
            reactant_coeffs[i]
            * self.get_predicted_total_exchange_energy(system, model_name)
            for i, system in enumerate(reactants)
        ]
        product_energies = [
            product_coeffs[i]
            * self.get_predicted_total_exchange_energy(system, model_name)
            for i, system in enumerate(products)
        ]
        return sum(product_energies) - sum(reactant_energies)
