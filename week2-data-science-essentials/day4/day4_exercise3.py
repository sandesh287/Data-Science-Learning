# Drop columns with more than 50% missing values

import pandas as pd
import numpy as np

# create a sample dataset
data = {
  "Name": ["Alice", "Bob", np.nan, "David", "Elen"],
  "Age": [25, np.nan, 30, 35, 28],
  "Score": [85, 90, np.nan, 88, 80],
  "Address": ["Kathmandu", np.nan, np.nan, "Lalitpur", np.nan]
}

df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# drop column using column name
df_column_name = df.drop("Age", axis=1)

print("Dataset after dropping columns by column_name:\n", df_column_name)

# drop column with more than 50% missing values
# calculate the threshold for non-missing values (50% of the number of rows)
threshold = len(df) * 0.5

# drop column where number of non-missing values is less than the threshold
# axis=1 specifies that we are operating on columns
df_final_cleaned = df.dropna(axis=1, thresh=threshold)

print("Dataset after dropping columns with >50% missing values:\n", df_final_cleaned)