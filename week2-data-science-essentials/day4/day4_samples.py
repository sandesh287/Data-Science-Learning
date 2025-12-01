import pandas as pd

# datasets as DataFrame
df = pd.DataFrame({
  "ID": [1, 2, 3],
  "Name": ["Alice", "Bob", "Charlie"],
  "Age": [25, 30, 35],
})

df1 = df
df2 = df

# Handle Missing values
# Drop missing values

# drop rows with missing values
df = df.dropna()
# drop columns with missing values
df = df.dropna(axis=1)

# Fill missing values
# replaces anything that is empty with 0, for that particular column
df["column_name"] = df["column_name"].fillna(0)

# Forward fill
df.fillna(method="ffill")
# Backward fill
df.fillna(method="bfill")

# Interpolation
# add values that are similar to other values
df["column_name"] = df["column_name"].interpolate()


# Data Transformation
# Renaming column
df = df.rename(columns={"old_name": "new_name"})

# Changing data types
df["column_name"] = df["column_name"].astype("float")  # convert to float type
df["column_name"] = pd.to_datetime(df["column_name"])  # convert to datetime type

# creating or modifying columns
df["new_column"] = df["existing_column"] * 2


# Combining and Merging DataFrames
# concatenation
# combine two DataFrames df1 and df2 along rows
combined = pd.concat([df1, df2], axis=0)
# combine two DataFrames df1 and df2 along columns
combined = pd.concat([df1, df2], axis=1)

# Merging
# based on key or conditions
merged = pd.merge(df1, df2, on="common_column") # merges df1 and df2 onto "common_column"
merged = pd.merge(df1, df2, how="left", on="common_column") # left join
merged = pd.merge(df1, df2, how="inner", on="common_column") # inner join

# Joining
# using index alignment
joined = df1.join(df2, how="inner")


