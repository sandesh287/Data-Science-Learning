# Statistical Analysis of Real-World Data
# 1. Perform Exploratory Data Analysis(EDA): Inspect dataset structure, summarize key features, visualize distributions, relationships and correlations. Plot and visualize relationships on tips dataset
# 2. Conduct Hypothesis Tests: Comparing tips by gender(t-test), test for independence between smoking status and time of the day(chi-square test)
# 3. Apply Linear Regression: Analyze relationship between total_bill and tip and interpret the slope, intercept and r-squared. Visualize the regression line on scatter plot
# 4. Extend the project by exploring additional relationship (eg. day of the week vs tip amount)
# 5. Perform multiple linear regression with additional variable (eg. include smoking status)

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
print('-------------------------------------------------------------------')


# Plot Regression (Scatterplot)
sns.scatterplot(x=df['total_bill'], y=df['tip'], color='blue')
plt.plot(df['total_bill'], model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression: Total Bill vs Tip (Scatterplot)')
plt.legend()
plt.show()



# IV. Explore additional relationship (tip vs day)
# Dependent variable: tip (numerical)
# Independent variable: day (categrical); Categories: Thur, Fri, Sat, Sun (more than 2)
# Hence, we are using one-way ANOVA test because we are comparing mean tip amount across multiple categories (days)

# One-way ANOVA
# libraries
from scipy.stats import f_oneway


# Group tips by day
groups = [
  df[df['day'] == day]['tip']
  for day in df['day'].unique()
]


# Perform One-way ANOVA
f_stat, p_val = f_oneway(*groups)

print('\n')

print('F-Statistic: ', f_stat)
print('P-Value: ', p_val)


# Interpretation
alpha = 0.05

print('\n')
print('-------------------------------------------------------------------')
print('\n')

if p <= alpha:
  print('Reject Null hypothesis: Mean tip amount is the same across all days')
else:
  print('Fail to reject Null hypothesis: At least one day has different mean tip')

print('\n')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')


# Visualization (boxplot)
sns.boxplot(x='day', y='tip', data=df)
plt.title('One-way ANOVA: Tip amount by Day of week')
plt.show()



# V. Multiple Linear Regression additing smoking status
# libraries
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D


# Encode smoker column (Yes/No -> 1/0)
df['smoker_encoded'] = LabelEncoder().fit_transform(df['smoker'])
 

# Prepare variables
X = df[['total_bill', 'smoker_encoded']]  # independent variables
y = df['tip']  # dependent variable


# Fit multiple linear regression
multi_model = LinearRegression()
multi_model.fit(X, y)


# Display coefficients
print('\n')

print("--- Multiple Linear Regression Results ---")
print("Slope Coefficients:")
print(f"  total_bill slope coefficient: {multi_model.coef_[0]:.4f}")
print(f"  smoker slope coefficient (1 = Yes): {multi_model.coef_[1]:.4f}")
print("Intercept:", multi_model.intercept_)
print("R-Squared:", multi_model.score(X, y))

print('\n')
print('-------------------------------------------------------------------')
print('-------------------------------------------------------------------')


# Create Regression Plane
bill_range = np.linspace(df['total_bill'].min(), df['total_bill'].max(), 50)
smoker_range = np.array([0, 1])  # categorical (0 = No, 1 = Yes)

BILL, SMOKER = np.meshgrid(bill_range, smoker_range)

# Convert DataFrame with proper column names
plane_df = pd.DataFrame({
  'total_bill': BILL.ravel(),
  'smoker_encoded': SMOKER.ravel()
})

Z = multi_model.predict(plane_df).reshape(BILL.shape)


# 3D Plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Scatterplot
ax.scatter(df['total_bill'], df['smoker_encoded'], df['tip'], color='blue', alpha=0.6, label='Data Points')

# Regression plane
ax.plot_surface(BILL, SMOKER, Z, color='red', alpha=0.3)

ax.set_xlabel('Total Bill')
ax.set_ylabel('Smoker (0 = No, 1 = Yes)')
ax.set_zlabel('Tip')
ax.set_title('3D Multiple Linear Regression: Tip ~ Total Bill + Smoker')

plt.legend()
plt.show()