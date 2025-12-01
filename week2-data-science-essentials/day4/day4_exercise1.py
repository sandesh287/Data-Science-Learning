# Clean a dataset by handling missing values and renaming columns

import pandas as pd
import numpy as np

# create a sample dataset
data = {
  "Name": ["Alice", "Bob", np.nan, "David"],
  "Age": [25, np.nan, 30, 35],
  "Score": [85, 90, np.nan, 88],
}

# convert dictionary dataset into DataFrome
df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# Filling missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Score"] = df["Score"].interpolate()

print("Filled Age and Score:\n", df)

# rename columns' name
df = df.rename(columns={"Name": "Student_Name", "Score": "Exam_Score"})
print("Renamed column:\n", df)