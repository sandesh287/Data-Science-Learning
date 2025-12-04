# Compute the expectation and variance of a weighted die (biased probabilities)
# example of biased probabilities (must sum to 1)
# Let's assume a die has outcomes 1 through 6, but the probabilities are not equal
# probs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]

# importing libraries
import numpy as np

# Die outcomes
outcomes = np.array([1, 2, 3, 4, 5, 6])

# Probabilities (must sum to 1)
probabilities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]

# Expectation (mean) of weighted die
expectation = np.sum(outcomes * probabilities)
print('Expectation (Mean): ', expectation)

# Variance of weighted die
variance = np.sum((outcomes - expectation)**2 * probabilities)
print('Variance: ', variance)


# Simulating the weighted Die to verify mean and variance
rolls = np.random.choice(outcomes, size=100000, p=probabilities)

print("Simulated Mean:", rolls.mean())
print("Simulated Variance:", rolls.var())