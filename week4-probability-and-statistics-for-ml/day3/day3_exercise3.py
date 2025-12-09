# Create confidence intervals for other statistics (eg. variance)

# importing libraries
import numpy as np
import pandas as pd
from scipy.stats import chi2

# load iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print(df.info())

# choosing one numeric column
data = df['petal_length']

# sample statistics
n = len(data)
sample_variance = np.var(data, ddof=1)  # unbiased sample variance
alpha = 0.05  # for 95% confidence interval

# Confidence Interval for variance using Chi-square distribution
chi2_lower = chi2.ppf(alpha / 2, n - 1)
chi2_upper = chi2.ppf(1 - alpha / 2, n - 1)

lower_bound = (n - 1) * sample_variance / chi2_upper
upper_bound = (n - 1) * sample_variance / chi2_lower

# results
print("Dataset: Iris (petal_length)")
print(f"Sample Variance: {sample_variance:.4f}")
print("95% Confidence Interval for Variance:")
print(f"Lower Bound: {lower_bound:.4f}")
print(f"Upper Bound: {upper_bound:.4f}")