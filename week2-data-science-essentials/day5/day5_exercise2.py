# calculate summary statistics for grouped data
# Objective: To use grouping and aggregation function to calculate summary statistics
# group data by categorical column, calculate statistics for numerical columns

import pandas as pd
import numpy as np

# sample dataset
data = {
  "Class": ["A", "B", "A", "B", "C", "C"],
  "Scores": [85, 90, 88, 72, 95, 80],
  "Age": [15, 16, 15, 17, 16, 15],
}

# create dataframe from dictionary
df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# grouped based on Class and calculate the mean
grouped = df.groupby("Class").mean()

# summary statistics for group data
stats = df.groupby("Class").agg(
  {"Scores": ["mean", "max", "min"], "Age": ["mean", "max", "min"]}
)

print(stats)