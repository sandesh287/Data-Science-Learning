# Simulate random variables from custom distributions (Eg. truncated normal distribution)
# Truncated normal distribution: it's a normal distribution restricted to a specific range. scipy provide truncnorm function
# a and b are standardized lower and upper bounds: a=(lower−μ)/σ​ , b=(upper−μ)/σ​


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Parameters for truncated normal
mu = 0  # mean
sigma = 1  # standard deviation
lower = -1  # lower bound
upper = 2  # upper bound

# Compute the 'a' and 'b' parameters for truncnorm
# These are standardized limits: (limit - mean)/std
a, b = (lower - mu) / sigma, (upper - mu) / sigma

# Simulate random variables
n_samples = 10000
samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_samples)  # generates random samples

# Visualize the distribution
x = np.linspace(lower - 0.5, upper + 0.5, 500)
pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)  # gives pdf for plotting

plt.figure(figsize=(8,5))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='skyblue', label='Simulated')
plt.plot(x, pdf, 'r', lw=2, label='Theoretical PDF')
plt.title('Truncated Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()