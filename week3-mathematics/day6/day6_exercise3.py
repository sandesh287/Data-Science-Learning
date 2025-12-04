# Visualize the distribution of data and highlight mean, median and mode using Matplotlib

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# sample data
data = [12, 15, 12, 15, 18, 19, 20, 20, 20, 22, 22, 23, 24, 25, 25, 25, 25, 30, 30, 35]

# calculate statistics
mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=False).mode  # using keepdims=False to get scalar

# create histogram
plt.figure(figsize=(10,6))
plt.hist(data, bins=10, color='skyblue', edgecolor='black', alpha=0.7)

# Highlight mean, median, mode
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val}')
plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val}')
plt.axvline(mode_val, color='orange', linestyle='-', linewidth=2, label=f'Mode: {mode_val}')

# Adding labels and title
plt.title('Data Distribution with Mean, Median and Mode')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()