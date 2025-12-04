# Simulate flipping coin 10,000 times and calculate probabilities of heads/tails

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Simulating 10,000 coin flips
flips = np.random.choice(['H', 'T'], size=10000)

# Count heads and tails
heads = np.sum(flips == 'H')
tails = np.sum(flips == 'T')

# Probabilities
p_heads = heads / len(flips)
p_tails = tails / len(flips)

print("Heads: ", heads, "-> Probability: ", p_heads)
print("Tails: ", tails, "-> Probability: ", p_tails)

# Visualization
plt.bar(['Heads', 'Tails'], [p_heads, p_tails])
plt.title("Probability of coin flip outcomes (10000 flips)")
plt.ylabel("Probability")
plt.show()