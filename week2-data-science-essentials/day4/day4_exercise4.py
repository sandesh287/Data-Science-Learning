# Merge three datasets and analyze relationships between them

import pandas as pd
import numpy as np

# creating a dataset dictionary and directly converting to DataFrame
df1 = pd.DataFrame({
  "customer_id": [1, 2, 3],
  "Name": ["Alice", "Bob", "Charlie"],
  "Age": [25, 30, 35],
})

df2 = pd.DataFrame({
  "customer_id": [1, 2, 3],
  "order_id": ["A1", "B1", "C1"],
})

df3 = pd.DataFrame({
  "order_id": ["A1", "B1", "C1"],
  "product": ["Laptop", "Mouse", "Keyboard"],
  "quantity": [3, 5, 7]
})

print("Dataset 1:\n", df1)
print("Dataset 2:\n", df2)
print("Dataset 3:\n", df3)

# merging first and second
merged_temp = pd.merge(df1, df2, how="left", on="customer_id")
print("Merged Dataset:\n", merged_temp)

final_merged = pd.merge(merged_temp, df3, how="inner", on="order_id")
print("Final Merged Dataset:\n", final_merged)

# Chooing how parameters:
# 'inner': returns only rows where the merge key exists in all merged DataFrames
# 'left': returns all rows from the left DataFrame and matching rows from right DataFrame
# 'right': returns all rows from the right DataFrame and matching rows from left DataFrame
# 'outer': returns all rows when there is a match in either left or right DataFrame


# Analyzing Relationship
print("Correlation Matrix:\n", final_merged.corr(numeric_only=True))

print("Average product quantity per customer:\n", final_merged.groupby("customer_id")["quantity"].mean())