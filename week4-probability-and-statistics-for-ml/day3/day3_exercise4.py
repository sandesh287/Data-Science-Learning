# Perform stratified sampling and compare intervals across strata
# Population: Iris dataset, Strata: species, Statistic: Mean of sepal_length, Method: Stratified sampling + Normal-based CI
# Statified sampling is essential when: Classes are meaningful(classification tasks), Data is heterogeneous
# CI comparison helps: Detect feature separation, Validate assumptions before modeling


# importing libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Stratified sampling by species
np.random.seed(42)
sample_fraction = 0.5  # 50% from each stratum

stratified_sample = (
  df.groupby('species', group_keys=False)
    .apply(lambda x: x.sample(frac=sample_fraction))
)

# Function to compute CI for mean
def confidence_interval_mean(data, confidence=0.95):
  n = len(data)
  mean = np.mean(data)
  std = np.std(data, ddof=1)
  z_value = norm.ppf((1 + confidence) / 2)
  margin_of_error = z_value * (std / np.sqrt(n))
  return mean, mean - margin_of_error, mean + margin_of_error

# Compute CI per stratum
results = []

for species, group in stratified_sample.groupby('species'):
  mean, lower, upper = confidence_interval_mean(group['sepal_length'])
  
  results.append({
    'Species': species,
    'Sample Size': len(group),
    'Mean': round(mean, 3),
    'CI Lower': round(lower, 3),
    'CI Upper': round(upper, 3)
  })
  
results_df = pd.DataFrame(results)

# Display results
print("Confidence Intervals by Species (Stratified Sampling)")
print(results_df)