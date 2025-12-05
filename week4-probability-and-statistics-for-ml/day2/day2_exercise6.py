# Explore datasets with real-world applications of distributions (Eg. stock prices, etc.) multiple datasets
# Done for 1. Stock Price (Apple (AAPL)), 2. Income distribution from tips dataset, 3. Weather data (Daily temperature) from flights dataset, 4. Housing prices (Boston housing)


# importing libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.datasets import fetch_california_housing

sns.set_theme(style='whitegrid')


# 1. Stock Prices: Apple (AAPl)
df_stock = yf.download('AAPL', start='2020-01-01', end='2024-12-31', auto_adjust=False)
df_stock['return_log'] = np.log(df_stock['Close'] / df_stock['Close'].shift(1))
returns_stock = df_stock['return_log'].dropna()


# 2. Income Data: Tips dataset (simulate income)
df_income = sns.load_dataset('tips')
income = df_income['total_bill']   # 'total_bill' as income proxy


# 3. Weather Data: Flights Dataset
df_weather = sns.load_dataset('flights')
temperature = df_weather['passengers']  # 'passengers' column as a proxy for temperature/count (numeric data)


# 4. Housing Prices: California Dataset
housing = fetch_california_housing(as_frame=True)
df_housing = housing.frame
housing_prices = df_housing['MedHouseVal']  # Using 'MedHouseVal' as target


# Helper Function: print stats
def print_stats(data, title):
  print(f"Dataset: {title}")
  print("Mean:", np.mean(data))
  print("Median:", np.median(data))
  print("Std Dev:", np.std(data))
  print("Skewness:", skew(data, bias=False))
  print("Excess Kurtosis:", kurtosis(data, bias=False))
  print("-" * 50)
  print()
  
datasets = {
  "AAPL Daily Log Returns": returns_stock,
  "Income (Tips: total_bill)": income,
  "Monthly Passengers (Flights)": temperature,
  "California Housing Prices": housing_prices
}

# print summary stats
for name, data in datasets.items():
  print_stats(data, name)
  

# Plot all 4 distributions in 2x2 grid
plt.figure(figsize=(14,10))

for i, (name, data) in enumerate(datasets.items(), 1):
  plt.subplot(2, 2, i)
  sns.histplot(data, bins=30, kde=True, color='skyblue')
  plt.title(name)
  plt.xlabel("Value")
  plt.ylabel("Frequency / Density")
  
plt.tight_layout()
plt.show()