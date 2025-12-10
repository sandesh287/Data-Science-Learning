# Perform a Hypothesis Test
# perform one sample t-test to test if a mean of a dataset differs from a known value

# importing libraries
import numpy as np
from scipy.stats import ttest_1samp

# sample data
data = [12, 14, 15, 16, 17, 18, 19]

# Null hypothesis: mean = 15
population_mean = 15

# Perform t-test
t_stat, p_value = ttest_1samp(data, population_mean)

print("T-Statistic: ", t_stat)
print("P-value: ", p_value)

# Interpret Results
alpha = 0.05
if p_value <= alpha:
  print("Reject the Null hypothesis: significant difference")
else:
  print("Fail to reject the Null hypothesis: no significant difference")