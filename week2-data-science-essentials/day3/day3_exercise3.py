# Load a local Excel file and explore its structure

import pandas as pd

# load local excel dataset
# need to install openpyxl to read excel file (pip install openpyxl)
file_path = 'Practice-Data.xlsx'
df = pd.read_excel(file_path)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

selected_columns = df[["JE Code", "Store"]]
print(selected_columns)

filtered_rows = df[(df["Store"] == "Vienna") & (df["Country"] == "Austria")]
print(filtered_rows)