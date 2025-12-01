# Group Data by a Categorical Column
# load sample dataset with categorical column inside it, group data by categorical column, calculate mean

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
print("Grouped mean:\n", grouped)