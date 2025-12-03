# Create and visualize a multinomial distribution for multi-class data

# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# 1. Define multinomial parameters
np.random.seed(42)

classes = ['Class A', 'Class B', 'Class C', 'Class D']
p = np.array([0.15, 0.25, 0.40, 0.20])  # probabilities must be sum to 1
n = 50  # number of trials per experiment
experiments = 2000  # number of multinomial draws


# 2. Generate multinomial random samples
samples = np.random.multinomial(n=n, pvals=p, size=experiments)  # samples.shape = (2000, 4)


# 3. Compute expected & empirical stats
expected_counts = n * p
empirical_mean = samples.mean(axis=0)
empirical_var = samples.var(axis=0)

print("Expected counts: ", expected_counts)
print("Empirical mean: ", empirical_mean)
print("Empirical variance: ", empirical_var)


# 4. Visualization
# Bar plot: Expected vs Empirical Mean
plt.figure(figsize=(8,5))
x = np.arange(len(classes))
width = 0.35

plt.bar(x - width/2, expected_counts, width, label="Expected (n * p)")
plt.bar(x + width/2, empirical_mean, width, label="Empirical Mean")

plt.xticks(x, classes)
plt.xlabel("Classes")
plt.ylabel("Counts")
plt.title("Multinomial Distribution: Expected vs Empirical Mean")
plt.legend()
plt.show()

# Histogram for each class
for i, cls in enumerate(classes):
  plt.figure(figsize=(7,4))
  plt.hist(samples[:, i], bins=15, edgecolor="black")
  plt.title(f"Distribution of Counts for {cls}")
  plt.xlabel("Count in one experiment")
  plt.ylabel("Frequency across experiments")
  plt.show()