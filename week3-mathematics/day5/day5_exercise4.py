# Compare Gaussian and uniform distributions for continuous data

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform


# 1. Generate continuous data
np.random.seed(42)

# Gaussian distribution (mean=0, std=1)
gaussian_data = np.random.normal(loc=0, scale=1, size=5000)

# Uniform distribution between -3 and 3
uniform_data = np.random.uniform(low=-3, high=3, size=5000)


# 2. Plot both distributions
plt.figure(figsize=(12, 6))

plt.hist(gaussian_data, bins=40, density=True, alpha=0.6, label="Gaussian", edgecolor="black")
plt.hist(uniform_data, bins=40, density=True, alpha=0.6, label="Uniform", edgecolor="black")

plt.title("Gaussian vs Uniform Distribution (Histogram)", fontsize=14)
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()


# 3. Plot PDFs (theoritical curves)
x = np.linspace(-4, 4, 500)

plt.figure(figsize=(12, 6))

# PDF for Gaussian
plt.plot(x, norm.pdf(x, 0, 1), label="Gaussian PDF", linewidth=3)

# PDF for Uniform
plt.plot(x, uniform.pdf(x, loc=-3, scale=6), label="Uniform PDF", linewidth=3)

plt.title("Gaussian vs Uniform Distribution (PDF Comparison)", fontsize=14)
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()


# Print Statistical comparison
print("Gaussian Mean: ", np.mean(gaussian_data))
print("Gaussian Standard Deviation: ", np.std(gaussian_data))
print("Uniform Mean: ", np.mean(uniform_data))
print("Uniform Standard Deviation: ", np.std(uniform_data))