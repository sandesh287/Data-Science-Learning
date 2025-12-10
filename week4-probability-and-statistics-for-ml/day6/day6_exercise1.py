# Calculate Correlation between Features
# compute and visualize correlations in a particular dataset
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

del df['species']  # as species column is not numerical variable, we need to delete this column

# compute correlation matrix
correlation_matrix = df.corr()

# plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()