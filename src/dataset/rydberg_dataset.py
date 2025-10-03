import numpy as np

from dataset.base_dataset import BaseDataset


class RydbergDataset(BaseDataset):
    def __init__(self, size: int = 1000):
        self.size = size
        np.random.seed(42)
        R_H = 1.097e7
        self.X = np.zeros((size, 2))
        self.y = np.zeros((size))
        for i in range(size):
            n_1 = np.random.randint(1, 7)
            n_2 = np.random.randint(n_1 + 1, 8)
            n_1 = float(n_1)
            n_2 = float(n_2)
            lambda_inv = R_H * ((1 / n_1**2) - (1 / n_2**2))
            self.X[i] = [n_1, n_2]
            self.y[i] = lambda_inv

    def get_data_train(self, sample_size=-1, shuffle=False):
        # Not implemented: sample_size or shuffle functionality
        return (
            self.X[0 : int(self.size * 0.75)],
            self.y[0 : int(self.size * 0.75)],
        )

    def get_data_test(self, sample_size=-1, shuffle=False):
        # Not implemented: sample_size or shuffle functionality
        return (
            self.X[int(self.size * 0.75) + 1, :],
            self.y[int(self.size * 0.75) + 1, :],
        )
