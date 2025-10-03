from dataset.rydberg_dataset import RydbergDataset
from regressor.symantic_regressor import SyMANTICRegressor


dataset = RydbergDataset(size=100)
regressor = SyMANTICRegressor(
    operators=["pow(2)", "^-1", "-", "/", "+"], metrics=[1e-8, 1.0]
)

X, y = dataset.get_data_train()
regressor.fit(X, y)
print(regressor.pred(X))
