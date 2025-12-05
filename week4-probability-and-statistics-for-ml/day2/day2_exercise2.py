# Analyze a Dataset's Distribution
# Objective: To analyze skewness and kurtosis of a dataset
# url: "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"


# importing libraries
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Analyze sepal_length
feature = df['sepal_length']
print("Skewness: ", skew(feature))
print("Kurtosis: ", kurtosis(feature))

# Visualize distribution
sns.histplot(feature, kde=True)
plt.title('Distribution of Sepal Length')
plt.show()