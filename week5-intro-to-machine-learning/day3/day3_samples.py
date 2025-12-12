# Advanced Regression Models: Polynomial Regression and Regularization

# Polynomial Regression

# libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# generate synthetic data
np.random.seed(42)

X = np.random.rand(100, 1) * 10
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

# transform features to polynomial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# fit the polymial regression
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.scatter(X, y_pred, color='red', label='Predicted Data')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE): ", mse)