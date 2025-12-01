# Just for the example
import pandas as pd

data = [1, 2, 3, 4]
df = pd.DataFrame(data)



# grouping using groupby()
grouped = df.groupby("column_name")

# Iterate over groups
for name, group in grouped:
  print(name)
  print(group)
  
# apply aggregation
grouped.mean()
grouped.sum()

# aggregation function
# using groupby()
df.groupby("category_column")["numeric_column"].mean()
df.groupby("category_column").agg({"numeric_column": ["mean", "max", "min"]})
# can do all at once. this is the power of groupby, when you combine grouping with aggregation

# using pivot_table()
# you can reshape the data with aggregation if you use pivot_table function
pivot = df.pivot_table(
  values="numeric_column",
  index="category_column",
  aggfunc="mean"
)

# using Custom aggregation
# you can apply custom function using agg function
def range_function(x):
  return x.max() - x.min()

# apply custom function using .agg()
df.groupby("category_column")["numeric_column"].agg(range_function)

# common statistics (mean, min, max)
df.groupby("category_column")["numeric_column"].mean()
df.groupby("category_column")["numeric_column"].max()
df.groupby("category_column")["numeric_column"].min()

# multi-aggregation
df.groupby("category_column").agg({"numeric_column": ["mean", "max", "min"]})