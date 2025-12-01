# importing matplotlib
import matplotlib.pyplot as plt

# basic plot
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

# plot the values in graph
plt.plot(x, y)
# show/display the plotted graph
plt.show()

# Line plot
# usually used to visualize trends over-time or sequences
plt.plot([1, 2, 3], [10, 20, 30], label="Trend", color="orange", linestyle="--", marker="x")
plt.title("Line Plot")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.legend()
plt.show()

# Bar Chart
# usually used for categorical data comparison
categories = ["A", "B", "C"]
values = [10, 15, 7]
plt.bar(categories, values, color="blue")
plt.title("Bar Chart")
plt.show()

# Histogram
# shows distribution of dataset
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.hist(data, bins=4, color="green", edgecolor="black")
plt.title("Histogram")
plt.show()

# Scatter plot
# visualizes the relation between two continuous variables
# used quite a lot in AI project
x = [1, 2, 3, 4, 5]
y = [10, 12, 25, 30, 45]
plt.scatter(x, y, color="red")
# Customizing Plots
plt.title("Scatter plot")
plt.xlabel("X-Axis Label")
plt.ylabel("Y-Axis Label")
plt.legend(["Dataset 1"])
plt.show()


# Seaborn
# importing
import seaborn as sns
import numpy as np
import pandas as pd

# random dataset
data = np.random.rand(5, 5)

# Heatmap
# visualizes matrix of data
sns.heatmap(data, annot=True, cmap="coolwarm")
plt.title("Heatmap")
plt.show()

# Pairplot
# works with dataframe
df = pd.DataFrame(data)
sns.pairplot(df)
plt.show()