# Probability Theory and Random Variables

# importing libraries
from itertools import product

# sample space of a dice roll
sample_space = list(range(1, 7))

# Probability of rolling an even number
even_numbers = [2, 4, 6]
P_even = len(even_numbers) / len(sample_space)
print("P(Even): ", P_even)


# Random variables, Expectation, Variance and Standard Deviation

# importing library
import numpy as np

# Random variable: dice roll
outcomes = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6] * 6)

# Expectation
expectation = np.sum(outcomes * probabilities)
print('Expectation (Mean): ', expectation)

# Variance and Standard Deviation
variance = np.sum((outcomes - expectation)**2 * probabilities)
std_dev = np.sqrt(variance)
print('Variance: ', variance)
print('Standard Deviation: ', std_dev)