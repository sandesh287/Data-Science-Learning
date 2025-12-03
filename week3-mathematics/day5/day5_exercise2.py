# Plot and Explore different Probability Distributions

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson


# Gaussian Distribution
x = np.linspace(-4, 4, 100)
y = norm.pdf(x, loc=0, scale=1)
plt.plot(x, y, label='Gaussian')
plt.title('Gaussian Distribution')
plt.show()


# Binomial Distribution
n, p = 10, 0.5
x = np.arange(0, n+1)
y = binom.pmf(x, n, p)
plt.bar(x, y, label='Binomial')
plt.title('Binomial Distribution')
plt.show()


# Poisson Distribution
lam = 3
x = np.arange(0, 10)
y = poisson.pmf(x, lam)
plt.bar(x, y, label='Poisson')
plt.title('Poisson Distribution')
plt.show()