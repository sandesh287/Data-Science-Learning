# Statistical Inference - Estimation and Confidence Interval

import numpy as np
from scipy.stats import norm, t

# sample data
data = [12, 14, 15, 16, 17, 18, 19]

# calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data, ddof=1)   # ddof=1 is for sample standard deviation

# 95% confidence interval (using t-distribution)
n = len(data)
t_value = t.ppf(0.975, df=n-1)
margin_of_error = t_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)
print(f"95% Confidence Interval: ({ci[0].item():.3f}, {ci[1].item()})")