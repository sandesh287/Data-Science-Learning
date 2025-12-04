# Create and Analyze random variables
# Objective: To create discrete(representing outcome of dice roll) and continuous(representing uniform distribution between 0 and 1) random variables and analyze them

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Discrete random variable: Dice roll
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6
plt.bar(outcomes, probabilities, color='blue', alpha=0.7)
plt.title("PMF of a Dice roll")
plt.xlabel("Outcomes")
plt.ylabel("Probability")
plt.show()

# Continuous random variable: Uniform distribution
x = np.linspace(0, 1, 100)
pdf = uniform.pdf(x, loc=0, scale=1)
plt.plot(x, pdf, color='red')
plt.title("PDF of uniform(0 - 1)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()