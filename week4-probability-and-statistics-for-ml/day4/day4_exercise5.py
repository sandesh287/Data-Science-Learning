# Perform hypothesis testing on proportions using binomial distribution

# importing libraries
import numpy as np
from scipy.stats import binom, norm

# Sample data
n = 200  # sample size
x = 30  # number of successes (clicks)
p0 = 0.10  # population proportion under H0

# sample proportion
p_hat = x / n
print("Sample Proportion: ", p_hat)

# Use normal approximation for large n (np0>5, n(1-p0)>5)
standard_error = np.sqrt(p0 * (1 - p0) / n)
z_stat = (p_hat - p0) / standard_error

# two-tailed p-value
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

# Results and Interpretation
print("Z-Statistic:", z_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Reject H0: The proportion is significantly different from 0.10")
else:
    print("Fail to reject H0: No significant difference in proportion")