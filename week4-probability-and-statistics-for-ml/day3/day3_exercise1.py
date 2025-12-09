# Calculate Confidence Intervals for sample data

# importing libraries
import numpy as np
from scipy.stats import norm, t

# Generate random sample data
data = np.random.normal(loc=50, scale=10, size=100)

# sample statistics
mean = np.mean(data)
std = np.std(data, ddof=1)
n = len(data)

# 95% confidence interval
z_value = norm.ppf(0.975)
margin_of_error = z_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print("Sample Mean: ", mean)
print(f"95% COnfidence Interval: ({ci[0].item()} , {ci[1].item()})")