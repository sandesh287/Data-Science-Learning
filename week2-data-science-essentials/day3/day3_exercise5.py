# Save filtered data to a new CSV file

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

filtered_rows = df[(df["sepal_length"] > 5.0) & (df["species"] == "setosa")]
# print("Filtered rows:\n", filtered_rows)

saved_file = "filtered.csv"
filtered_rows.to_csv(saved_file)
print(f"Successfully added data to {saved_file}")

filtered_rows_2 = df[(df["sepal_width"] > 3.7) & (df["species"] == "setosa")]
saved_file_2 = "filtered_2.csv"
filtered_rows_2.to_csv(saved_file_2)
print(f"Successfully added data to {saved_file_2}")