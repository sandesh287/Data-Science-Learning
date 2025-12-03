# Use probability distributions to simulate and analyze real-world datasets
# We simulate 3 different real-world scenarios:
# - Customer arrivals per hour → Poisson distribution
# - Product weights in a factory → Gaussian (Normal) distribution
# - Click-through probability of ads → Bernoulli/Binomial distribution


# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# 1. Simulate Customer Arrivals (Poisson Distribution)
# Poisson is commonly used for modeling count-based events (call/hour, arrivals/hour, errors/sec)

np.random.seed(42)

# Average 12 customers arrive per hour
lambda_rate = 12
hours = 200

customer_arrivals = np.random.poisson(lam=lambda_rate, size=hours)

plt.figure(figsize=(8,4))
plt.hist(customer_arrivals, bins=15, edgecolor="black", density=True)
plt.title("Simulated Customer Arrivals per Hour (Poisson Distribution)")
plt.xlabel("Customers per hour")
plt.ylabel("Frequency")
plt.show()

print("Average simulated arrivals: ", customer_arrivals.mean())


# 2. Simulate Product Weights (Normal Distribution)
# Factories often assume weight/size follows a Gaussian distribution

weights = np.random.normal(loc=500, scale=20, size=5000)

plt.figure(figsize=(8,4))
plt.hist(weights, bins=40, edgecolor="black", density=True)
plt.title("Simulated Product Weights (Normal Distribution)")
plt.xlabel("Weight (grams)")
plt.ylabel("Density")
plt.show()

print("Mean weight: ", weights.mean())
print("Standard deviation: ", weights.std())


# 3. Simulate Ad Click-Through Rate (Binomial Distribution)
# Each ad shown has a probability p of getting clicked

# show ad to 100 users, click probability = 5%
p = 0.05
n_users = 100

clicks = np.random.binomial(n=n_users, p=p, size=500)

plt.figure(figsize=(8,4))
plt.hist(clicks, bins=20, edgecolor="black", density=True)
plt.title("Ad Click Simulation (Binomial Distribution)")
plt.xlabel("Clicks per 100 users")
plt.ylabel("Frequency")
plt.show()

print("Average clicks per 100 users: ", clicks.mean())