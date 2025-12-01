# Convert categorical data to numerical using one-hot encoding
# one-hot encoding is a technique to convert categorical data into numerical format suitable for ML algorithms and can achieved using pd.get_dummies() function

import pandas as pd
import numpy as np

# creating sample dataset with categorical data in DataFrame
df = pd.DataFrame({
  'ID': [1, 2, 3, 4, 5],
  'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
  'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
  'Price': [10, 20, 30, 15, 25],
})

print("Original Dataset:\n", df)

# apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Color', 'Size'])
# The columns argument specifies which columns to encode. If omitted, get_dummies() will automatically encode all object-type columns
# pd.get_dummies() automatically identifies the unique categories within 'Color' and 'Size' columns and creates binary columns for each category.
print("DataFrame after one-hot encoding:\n", df_encoded)

# To drop the first category of each encoded column, which can help prevent multicollinearity in some models.
df_encoded_drop_first = pd.get_dummies(df, columns=['Color', 'Size'], drop_first=True)
print("DataFrame after one-hot encoding with drop_first=True:\n", df_encoded_drop_first)
# drop_first=True : it removes the first category of each encoded column, which can help prevent multicollinearity in some ML models