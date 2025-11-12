import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from dataset.rydberg_dataset import RydbergDataset
from dataset.exx_dataset import ExxDataset
from regressor.symantic_regressor import SyMANTICRegressor
from regressor.krr_regressor import KRRRegressor
from integrator.exx_integrator import ExxIntegrator
from integrator.model_gx_integrator import ModelGxIntegrator

# dataset = RydbergDataset(size=100)
# regressor = SyMANTICRegressor(
#     operators=["pow(2)", "^-1", "-", "/", "+"], metrics=[1e-8, 1.0]
# )
# regressor = SyMANTICRegressor(
#     scaler=MaxAbsScaler(),
#     operators=["+", "-", "*", "/", "exp", "sin", "cos"],
#     metrics=[1e-8, 1.0],
# )



dataset = ExxDataset(system_type="molecules")
np.random.seed(1)


# ----- Train -----
# regressor = KRRRegressor()
# print("loading training data")
# X_train, y_train = dataset.get_data_train(sample_size=3000, shuffle=True)
# print("loading testing data")
# X_test, y_test = dataset.get_data_test(sample_size=3000, shuffle=True)
# print("fitting model")
# regressor.fit(X_train, y_train)
# print("completed fitting")

# regressor.save()
# train_pred = regressor.pred(X_train)
# test_pred = regressor.pred(X_test)
# print(f"Training MSE: {np.mean((train_pred - y_train)**2)}")
# print(f"Testing MSE: {np.mean((test_pred - y_test)**2)}")
# ----- / Train -----


# ----- Apply to Integrator -----
regressor = KRRRegressor()
regressor.load()
print('Creating integrator')
integrator = ExxIntegrator('molecules')
print(dataset.systems)
print('calculating energy')
system = 'NO2'
print(system)
energy = integrator.get_total_energy(system)
print(energy)

print('Creating model integrator')
integrator = ModelGxIntegrator(regressor, 'molecules')
print('calculating energy')
energy = integrator.get_total_energy(system)
print(energy)
# ----- / Apply to Integrator -----
