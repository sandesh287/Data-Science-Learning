# Load Dataset and Select specific Columns and Filter Rows

import pandas as pd

# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# select specific columns
# It can be used for data modeling where you can take all the different columns except for the ones you interested in to predict and train the model with that
selected_columns = df[["species", "sepal_length"]]
print("Selected Coloumn:\n", selected_columns)

# Filter rows
filtered_rows = df[(df["sepal_length"] > 5.0) & (df["species"] == "setosa")]
print("Filtered Rows:\n", filtered_rows)