# Correlation and Regression Analysis

# Correlation
# libraries
import numpy as np
from scipy.stats import pearsonr, spearmanr

# dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Pearson Correlation
r, _ = pearsonr(x, y)  # dont need another, so gave _
print("Pearson Correlation Coefficient: ", r)

# Spearman Correlation
rho, _ = spearmanr(x, y)
print("Spearman Correlation Coefficient: ", rho)


# Linear Regression
from sklearn.linear_model import LinearRegression
import numpy as np

# sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 8, 10])

# fit linear regression
model = LinearRegression()
model.fit(x, y)  # help getting slope, intercept and R-squared

print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared: ", model.score(x, y))