import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from dataset.exx_dataset import ExxDataset
from regressor.krr_regressor import KRRRegressor
from pipeline.exx_pipeline import ExxPipeline

np.random.seed(1)

# define a dataset
dataset = ExxDataset(system_type="molecules") # system_type could be None (all) or any of 'bulks', 'molecules', or 'cubic_bulks'

print("Available systems:", dataset.get_available_systems())
print("Atoms in NH2OH", dataset.get_atoms_in_system("NH2OH"))

# define a regressor
regressor = KRRRegressor(alpha=0.0001, gamma=0.1)  # best parameters used in Jagriti's Thesis

# create a pipeline instance
pipeline = ExxPipeline(dataset=dataset, regressor=None)

# Train a model
pipeline.set_regressor(regressor)  # you can optionally just pass this directly into the pipeline constructor above
pipeline.train_regressor(sample_size=3000, shuffle_data=True, model_name="example_model.pkl")
true_exchange_energy = pipeline.get_true_total_exchange_energy("NH2OH")
pred_exchange_energy = pipeline.get_predicted_total_exchange_energy("NH2OH", model_name="example_model.pkl")

print("True Exchange Energy:", true_exchange_energy, "Ha")
print("Predicted Exchange Energy:", pred_exchange_energy, "Ha")
print("Error:", true_exchange_energy - pred_exchange_energy, "Ha")

# Calculate formation energies
true_formation_exchange_energy = pipeline.get_true_reaction_exchange_energy(reactants=["NH3", "NO", "H2O"], products=["NH2OH"])
predicted_formation_exchange_energy = pipeline.get_predicted_reaction_exchange_energy(reactants=["NH3", "NO", "H2O"], products=["NH2OH"], model_name="example_model.pkl")

print("True Exchange Energy:", true_formation_exchange_energy, "Ha")
print("Predicted Exchange Energy:", predicted_formation_exchange_energy, "Ha")
print("Error:", true_formation_exchange_energy - predicted_formation_exchange_energy, "Ha")
