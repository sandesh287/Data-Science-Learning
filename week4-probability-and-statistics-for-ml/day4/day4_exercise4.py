# Use the Iris dataset to test if the mean sepal length differs between two species

# importing libraries
import pandas as pd
from scipy.stats import ttest_ind

# Load iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Select two large groups
setosa = df[df['species'] == 'setosa']['sepal_length']
versicolor = df[df['species'] == 'versicolor']['sepal_length']

# Perform two-sample t-test (two-tailed)
t_stat, p_value = ttest_ind(setosa, versicolor, equal_var=False)  # Welch's t-test

# Print results
print("Setosa Mean: ", setosa.mean())
print("Versicolor Mean: ", versicolor.mean())
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Reject H0: The means are significantly different")
else:
    print("Fail to reject H0: No significant difference between means")