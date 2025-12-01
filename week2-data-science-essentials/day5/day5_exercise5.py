# create a custom aggregation function to calculate the variance for each group

import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("sales_data.csv")

# Function to calculate variance
def calculate_variance(data, sample=True):
  n = len(data)
  if n == 0:
    raise ValueError("Input List cannot be empty!")
  if sample and n < 2:
    raise ValueError("Sample variance requires at least two data points.")
  
  mean = sum(data) / n
  sqaured_difference = [(x - mean) ** 2 for x in data]
  
  if sample:
    variance = sum(sqaured_difference) / (n - 1)
  else:
    variance = sum(sqaured_difference) / n
    
  return variance

# custom aggregation for customized function
custom_aggregation = df.groupby("Region")["Sales_Amount"].agg(calculate_variance)
print(custom_aggregation)

variance_aggregation = df.groupby("Region")["Sales_Amount"].var()
print(variance_aggregation)