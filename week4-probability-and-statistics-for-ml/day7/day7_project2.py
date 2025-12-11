# Statistical Analysis of Real-World Data
# 1. Perform Exploratory Data Analysis(EDA): Inspect dataset structure, summarize key features, visualize distributions, relationships and correlations. Plot and visualize relationships on tips dataset
# 2. Conduct Hypothesis Tests: Comparing tips by gender(t-test), test for independence between smoking status and time of the day(chi-square test)
# 3. Apply Linear Regression: Analyze relationship between total_bill and tip and interpret the slope, intercept and r-squared. Visualize the regression line on scatter plot

# Dataset url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"


# I. Perform EDA
# libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)


# Inspect data
print(df.info())

print('\n')
print('-------------------------------------------------------------------')
print('\n')

print(df.describe())

print('\n')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('\n')


# Visualize Distributions
sns.histplot(df['total_bill'], kde=True)
plt.title('Distribution of total_bill (Histplot)')
plt.show()


# need to delete non-numerical columns to show data in heatmap
del df['sex']
del df['smoker']
del df['day']
del df['time']

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# II. Conduct hypothesis Testing
# libraries
from scipy.stats import chi2_contingency, ttest_ind


# Load dataset, as we had to delete non-numerical columns for above operations
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)


# 1. Chi-Square Test
# Contingency table
contingency_table = pd.crosstab(df['smoker'], df['time'])


# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print('Chi-Square Statistic: ', chi2)
print('P-Value: ', p)

print('\n')
print('-------------------------------------------------------------------')
print('\n')


# Interpret result
alpha = 0.05
if p <= alpha:
  print('Reject null hypothesis: Variables are dependent')
else:
  print('Fail to reject null hypothesis: Variables are independent')

print('\n')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')
print('\n')


# 2. T-test
# separate data by gender
male_tips = df[df['sex'] == 'Male']['tip']
female_tips = df[df['sex'] == 'Female']['tip']


# Perform t-test
t_stat, p_value = ttest_ind(male_tips, female_tips)
print('T-Statistic: ', t_stat)
print('P-Value: ', p_value)

print('\n')
print('-------------------------------------------------------------------')
print('\n')


# Interpret results
alpha = 0.05
if p <= alpha:
  print('Reject null hypothesis: Significant difference')
else:
  print('Fail to reject null hypothesis: No significant difference')

print('\n')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')



# III. Apply Linear Regression
# libraries
from sklearn.linear_model import LinearRegression
import numpy as np


# Define variables
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values


# Fit linear regression
model = LinearRegression()
model.fit(X, y)


# print the coefficients
print('\n')

print('Slope: ', model.coef_[0])
print('Intercept: ', model.intercept_)
print('R-Squared: ', model.score(X, y))

print('\n')
print('-------------------------------------------------------------------')


# Plot Regression (Scatterplot)
sns.scatterplot(x=df['total_bill'], y=df['tip'], color='blue')
plt.plot(df['total_bill'], model.predict(X), color='red', label='Regression Line')
plt.title('Total Bill vs Tip (Scatterplot)')
plt.legend()
plt.show()