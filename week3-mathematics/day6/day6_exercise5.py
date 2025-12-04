# Calculate confidence intervals for proportions in a dataset
# Suppose we have a survey dataset, and we want to calculate the CI for the proportion of people who said 'Yes'


# importing libraries
import numpy as np


# sample data
responses = ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']


# count successes ('Yes')
successes = np.sum(np.array(responses) == 'Yes')
n = len(responses)
p_hat = successes / n
print('Sample Proportion: ', p_hat)


# Use normal approximation for CI
# For large samples, the 95% CI for a proportion is: CI = p ± Z ⋅ sqrt((p(1−p))/n​), where Z = 1.96 for 95% confidence
# Confidence level
Z = 1.96  # 95% confidence

# Standard Error
SE = np.sqrt(p_hat * (1 - p_hat) / n)

# Confidence Interval
ci_lower = p_hat - Z * SE
ci_upper = p_hat + Z * SE

print(f'95% CI for proportion: ({ci_lower: .3f}, {ci_upper: .3f})')



# Using statsmodels for real datasets
import statsmodels.api as sm

# CI using statsmodels
ci_low, ci_upp = sm.stats.proportion_confint(count=successes, nobs=n, alpha=0.05, method='normal')
print(f"95% CI using (statsmodels): ({ci_low:.3f}, {ci_upp:.3f})")