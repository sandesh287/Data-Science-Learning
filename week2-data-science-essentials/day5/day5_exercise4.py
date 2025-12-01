# Use pivot_table to calculate total sales per region and per year

import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("sales_data.csv")

# using pivot_table
# sales per region
pivot = df.pivot_table(
  values="Sales_Amount",
  index="Region",
  aggfunc="mean"
)

print(pivot)

# sales per head
pivot_sales_Rep = df.pivot_table(
  values="Sales_Amount",
  index="Sales_Rep",
  aggfunc="mean"
)

print(pivot_sales_Rep)