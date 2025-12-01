# Combine multiple plots in a single figure using Matplotlib's subplot

# importing libraries
import matplotlib.pyplot as plt
import numpy as np

# create data for plotting
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = x**2

# create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

# plot an individual subplots using the axes array
axes[0, 0].plot(x, y1, color='blue')
axes[0, 0].set_title('Sine Wave')

axes[0, 1].plot(x, y2, color='red')
axes[0, 1].set_title('Cosine Wave')

axes[1, 0].plot(x, y3, color='green')
axes[1, 0].set_title('Tangent Wave')

axes[1, 1].plot(x, y4, color='purple')
axes[1, 1].set_title('Quadratic Function')

# adjust layout to prevent overlapping titles/labels
plt.tight_layout()

# display the figure
plt.show()