# Project - 1
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

# Project - 3: Use another real-world dataset (eg. healthcare or sales data) to apply similar techniques as project 2


# I. Performing EDA
# libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)


# Inspect data
print(df.info())
print(df.describe())


# Visualize distributions
sns.histplot(df['total_bill'], kde=True)
plt.title('Distribution of total_bill')
plt.show()


# need to drop / delete non-numerical column for heatmap
del df['sex']
del df['smoker']
del df['day']
del df['time']


# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# II. Conducting Hypothesis Testing
# libraries
from scipy.stats import ttest_ind


# Since we are using 'sex' column, we need to load dataset again, as we had deleted 'sex' column for heatmap
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# separate data by gender
male_tips = df[df['sex'] == 'Male']['tip']
female_tips = df[df['sex'] == 'Female']['tip']


# Perform t-test
t_stat, p_value = ttest_ind(male_tips, female_tips)
print('T-Statistic: ', t_stat)
print('P-Value: ', p_value)


# Interpret results
alpha = 0.05
if p_value <= alpha:
  print('Reject null hypothesis: Significant difference')
else:
  print('Fail to reject null hypothesis: No significant difference')
  
  

# III. Applying Linear Regression
# libraries
from sklearn.linear_model import LinearRegression
import numpy as np


# Define variables
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values


# Fit Linear regression
model = LinearRegression()
model.fit(X, y)


# Output the coefficients
print('Slope: ', model.coef_[0])
print('Intercept: ', model.intercept_)
print('R-Squared: ', model.score(X, y))


# Plot Regression
sns.scatterplot(x=df['total_bill'], y=df['tip'], color='blue')
plt.plot(df['total_bill'], model.predict(X), color='red', label='Regression Line')
plt.title('Total Bill vs Tip')
plt.legend()
plt.show()