# Load and Explore a Sample dataset

# https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

import pandas as pd

# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Explore the structure
print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())
print("Dataset Information:\n", df.info())
print("Statistical Information:\n", df.describe())