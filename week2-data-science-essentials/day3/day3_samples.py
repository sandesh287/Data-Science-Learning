import pandas as pd

# define Series
# Series: 1D labeled array capable of holding data of any type
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s)
# Output:  
#          a    10
#          b    20
#          c    30
#          dtype: int64

# DataFrame
# creates 2-dimension labeled data structure which looks like table
data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)
print(df)
# Output:     
#             Name  Age
#         0  Alice   25
#         1    Bob   30


# Common Data Loading Methods
# from CSV
df = pd.read_csv("data.csv")
# from excel
df = pd.read_excel("data.xls")
# from dictionary
data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}
df = pd.DataFrame(data)


# Saving Data
# saving CSV file
df.to_csv("data.csv", index=False)  # if don't want index, then index=False
# saving excel file
df.to_excel("data.xlsx")


# DataFrom Operations
# Viewing Data
print(df.head())  # prints first five rows of df dataset
print(df.tail(3))  # prints last 3 rows of df dataset
print(df.info())  # gives information summary of the dataFrame
print(df.describe())  # gives statistical summary of the dataFrame

# Selecting and Indexing
# select particular column
print(df["Name"])  # selects Name column
print(df[["Name", "Age"]])  # selects Name and Age column (multiple column)

# Filtering rows
print(df[df["Age"] > 25])  # filter rows where Age is greater than 25

# selecting by position
print(df.iloc[0])  # first row by position
print(df.iloc[:, 0])  # first column by position

# selecting by label
print(df.loc[0])  # first row by index label
print(df.loc[:, "Name"])  # Name column by label("Name")