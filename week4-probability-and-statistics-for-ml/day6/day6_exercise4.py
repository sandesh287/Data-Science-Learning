# Compare correlation and regression results for non-linear relationships

# libraries
import numpy as np  # numerical operations and array handling
import matplotlib.pyplot as plt  # plotting graphs
from scipy.stats import pearsonr  # compute Pearson correlation coefficient
from sklearn.linear_model import LinearRegression  # fit linear regression method
from sklearn.preprocessing import PolynomialFeatures  # xonvert x into polynomial form(x, x², ...)


# Generate non-linear data
np.random.seed(42)  # fixes random numbers, so results are reproducible

# reshape(-1,1): converts to a column vector (required by sklearn)
x = np.linspace(-5, 5, 100).reshape(-1, 1)  # creates 100 evenly spaced values from -5 to 5
y = x**2 + np.random.randn(100, 1) * 2  # create quadratic equation y=x²; np.random.randn() * 2: adds noise and multiply by 2


# Correlation
# flatten(): converts 2D array to 1D
# pearsonr(): computes linear correlation
# corr: correlation coefficient
# _: ignores the p-value
corr, _ = pearsonr(x.flatten(), y.flatten())
print('Pearson Correlation: ', corr)  # value close to 0 means no linear relationship


# Linear Regression
lin_model = LinearRegression()  # create a linear regression model object
lin_model.fit(x, y)  # fits the straight-line model, tries to model y=mx+c
y_lin_pred = lin_model.predict(x)  # uses fitted model to predict y, produces predicted straight-line values


# Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)  # specifies polynomial degree 2, will create features: [1, x, x²]
X_poly = poly.fit_transform(x)  # converts x into polynomial features, eg. x → [1, x, x²]

poly_model = LinearRegression()  # uses linear regression, but on polynomial features
poly_model.fit(X_poly, y)  # fits model to: y=a+bx+cx²
y_poly_pred = poly_model.predict(X_poly)  # predicts y using polynomial regression and produces curved fit


# Visualization
plt.figure(figsize=(9,5))
plt.scatter(x, y, color='blue', alpha=0.6, label='Data')
plt.plot(x, y_lin_pred, color='red', label='Linear Regression')
plt.plot(x, y_poly_pred, color='green', label='Polynomial Regression')
plt.title("Correlation vs Regression (Non-linear Relationship)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()