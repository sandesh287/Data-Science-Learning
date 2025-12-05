# Explore datasets with real-world applications of distributions (Eg. stock prices)


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import yfinance as yf


# 1. Fetch stock data: eg. Apple (AAPL)
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-12-31'

# Dataframe
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)


# 2. Compute daily returns
# Percentage returns
df['return_pct'] = df['Close'].pct_change()
# Log returns
df['return_log'] = np.log(df['Close'] / df['Close'].shift(1))

# Drop Nan values from first row
returns = df['return_log'].dropna()


# 3. Summary Statistics
print("Summary Statistics for Daily Log Returns:")
print("Mean log return:", returns.mean())
print("Std dev log return:", returns.std())
print("Skewness:", skew(returns, bias=False))
print("Excess Kurtosis:", kurtosis(returns, bias=False))


# 4. Visualize distribution
plt.figure(figsize=(8,5))
sns.histplot(returns, bins=100, kde=True, color='skyblue')
plt.title(f'{ticker} Daily log-return distribution \n(from {start_date} to {end_date})')
plt.xlabel('Log return')
plt.ylabel('Frequency / Density')
plt.grid(True)
plt.show()