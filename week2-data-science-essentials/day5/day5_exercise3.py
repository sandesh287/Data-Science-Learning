# Create a dataset of sales data and group it by region or product category

import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("sales_data.csv")
# print(df)

grouped = df.groupby("Region").mean("Sales_Amount")
print(grouped)

grouped_by_product_category = df.groupby("Product_Category").mean("Sales_Amount")
print(grouped_by_product_category)

stats = df.groupby("Region").agg(
  {
    "Sales_Amount": ["mean", "max", "min"], 
    "Quantity_Sold": ["mean", "max", "min"],
    "Unit_Cost": ["mean", "max", "min"],
    "Unit_Price": ["mean", "max", "min"],
  }
)

print(stats)