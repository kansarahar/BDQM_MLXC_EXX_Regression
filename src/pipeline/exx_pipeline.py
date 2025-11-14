import numpy as np
import sympy
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
