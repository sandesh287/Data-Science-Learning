# Create a Heatmap with Seaborn
# load dataset and calculate correlation matrix, visualize correlation matrix
# dataset
# https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# The "species" column got deleted, because it is not of numeric value, and only numeric value can be calculated for correlation matrix
# the command to delete column from dataset
del df['species']

# calculate correlation matrix
correlation_matrix = df.corr()

# plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()