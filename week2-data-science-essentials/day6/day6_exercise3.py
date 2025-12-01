# Create a histogram with multiple datasets overlaid

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sample datasets
data1 = np.random.normal(loc=0, scale=1, size=1000)
data2 = np.random.normal(loc=1, scale=1.5, size=1000)
data3 = np.random.normal(loc=-0.5, scale=0.8, size=1000)

# creating a dataframe
df = pd.DataFrame({'Dataset1': data1, 'Dataset2': data2, 'Dataset3': data3})

# plotting the overlaid histograms
# bins: number of intervals
# alpha: transparency
plt.hist(df["Dataset1"], bins=30, alpha=0.5, label='Dataset 1', color='blue')
plt.hist(df["Dataset2"], bins=30, alpha=0.5, label='Dataset 2', color='red')
plt.hist(df["Dataset3"], bins=30, alpha=0.5, label='Dataset 3', color='green')

# add labels, title and legend
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Overlaid Histograms of Multiple Datasets")
plt.legend()
plt.show()