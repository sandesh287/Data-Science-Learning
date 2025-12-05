# Compare the effects of skewness and kurtosis on different datasets


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Loading two real-world datasets from seaborn library
# Titanic dataset (passenger info)
titanic = sns.load_dataset('titanic')
# Tips dataset (passenger info)
tips = sns.load_dataset('tips')

# creating dictionary to easily iterate over datasets and their numeric columns
datasets = {
  'Titanic (age, fare)': titanic[['age', 'fare']],
  'Tips (total_bill, tip)': tips[['total_bill', 'tip']]
}

# calculating kurtosis and skewness
# Iterate over each dataset
for name, df in datasets.items():
  print('Dataset: ', name)
  # Iterate over each numeric column in the dataset
  for col in df.columns:
    # Drop missing values
    series = df[col].dropna()
    # compute skewness and excess kurtosis
    print(f'Column: {col}')
    print('Skewness: ', skew(series, bias=False))
    print('Excess Kurtosis: ', kurtosis(series, bias=False))
  print()  # blank line for readability


# Visualize distributions side-by-side
plt.figure(figsize=(12, 8))  # figure size for multiple subplots
i = 1  # subplot counter

# Iterate over datasets again for plotting
for name, df in datasets.items():
  for col in df.columns:
    plt.subplot(len(datasets), len(df.columns), i)  # row x col layout\
    # Histogram with KDE layout
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'{name}\n{col}')
    plt.xlabel(col)
    plt.ylabel('Count / Density')
    i += 1


plt.tight_layout()  # adjust spacing to avaid overlap
plt.show()