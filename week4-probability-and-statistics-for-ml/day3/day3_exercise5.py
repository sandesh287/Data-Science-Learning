# Visualize confidence intervals for multiple samples using Matplotlib

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Function to compute CI for mean
def confidence_interval_mean(data, confidence=0.95):
  n = len(data)
  mean = np.mean(data)
  std = np.std(data, ddof=1)
  z_value = norm.ppf((1 + confidence) / 2)
  margin_of_error = z_value * std / np.sqrt(n)
  return mean, margin_of_error

# Compute CI for species
species_names = []
means = []
errors = []

for species, group in df.groupby('species'):
  mean, margin_of_error = confidence_interval_mean(group['sepal_length'])
  species_names.append(species)
  means.append(mean)
  errors.append(margin_of_error)
  
# Visualize CI
plt.figure(figsize=(8, 5))
plt.errorbar(
    species_names,
    means,
    yerr=errors,
    fmt='o',
    capsize=10
)

plt.title("95% Confidence Intervals for Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()