# Create a DataFrame from a dictionary and add a new calculated column

import pandas as pd

# create a dictionary
data = {
  'Name': ['Alice', 'Bob', 'Charlie', 'David'],
  'Age': [25, 30, 35, 28],
  'Score1': [85, 90, 78, 92],
  'Score2': [70, 88, 95, 80]
}

# create the dataframe from the dictionary
df = pd.DataFrame(data)

# add a new calculated column 'Total_Score' which is sum of Score1 and Score2
df['Total_Score'] = df['Score1'] + df['Score2']

# adding more calculated column
# df['Average_Score'] = df['Total_Score'] / 2
# OR, to calculate average score
df['Average_Score'] = df[['Score1', 'Score2']].mean(axis=1)

df['Grade'] = df['Total_Score'].apply(lambda x: 'Excellent' if x >= 173 else 'Good')

print(df)