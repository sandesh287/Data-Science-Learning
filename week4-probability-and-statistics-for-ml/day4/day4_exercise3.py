# Perform z-test for large sample sizes

# importing libraries
import numpy as np
from scipy.stats import norm
import pandas as pd


# Generate random samples
np.random.seed(42)

# Sample 1: mean=50, std=5, size=1000
sample1 = np.random.normal(loc=50, scale=5, size=1000)

# Sample 2: mean=52, std=5, size=1000
sample2 = np.random.normal(loc=52, scale=5, size=1000)


# Compute sample statistics
n1 = len(sample1)
n2 = len(sample2)
mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
var1 = np.var(sample1, ddof=1)
var2 = np.var(sample2, ddof=1)


# Compute z-statistic
z_value = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)

# twio-tailed p-value (checks for any difference, not direction specific)
p_value = 2 * (1 - norm.cdf(abs(z_value)))


# Print results
print("Sample 1 Mean: ", mean1)
print("Sample 2 Mean: ", mean2)
print("Z-Statistic: ", z_value)
# Displaying p-value in scientific notation
print("P-Value (scientific notation): {:.2e}".format(p_value))

# Interpretation
if p_value < 0.05:
    print("Reject H0: The means are significantly different")
else:
    print("Fail to reject H0: No significant difference between means")