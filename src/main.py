import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from dataset.rydberg_dataset import RydbergDataset
from dataset.krr_exx_dataset import KRRExxDataset
from regressor.symantic_regressor import SyMANTICRegressor
from regressor.krr_regressor import KRRRegressor


# dataset = RydbergDataset(size=100)
# regressor = SyMANTICRegressor(
#     operators=["pow(2)", "^-1", "-", "/", "+"], metrics=[1e-8, 1.0]
# )
dataset = KRRExxDataset(system_type="molecules")
regressor = SyMANTICRegressor(
    scaler=MaxAbsScaler(),
    operators=["+", "-", "*", "/", "exp", "sin", "cos"],
    metrics=[1e-8, 1.0],
)

np.random.seed(1)

print("loading training data")
X_train, y_train = dataset.get_data_train(sample_size=30, shuffle=True)
print("loading testing data")
X_test, y_test = dataset.get_data_test(sample_size=30, shuffle=True)
print("fitting model")
regressor.fit(X_train, y_train)
print("completed fitting")

regressor.save()
train_pred = regressor.pred(X_train)
test_pred = regressor.pred(X_test)
print(f"Training MSE: {np.mean((train_pred - y_train)**2)}")
print(f"Testing MSE: {np.mean((test_pred - y_test)**2)}")
