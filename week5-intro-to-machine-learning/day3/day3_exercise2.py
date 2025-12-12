# Use Lasso and Ridge Regression
# Train Lasso and Ridge Regression model on California Housing Dataset and observe the effect of Regularization
# Using feature variable (median Income) and target variable (median House value), apply polynomial feature transformation degree 2, split the data into training and testing set, train Lasso & Ridge Regression Models with different regularization parameters and finally evaluating the model using MSE



# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load the california housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame


# Select feature (median income) and target (median house value)
X = df[['MedInc']]
y = df[['MedHouseVal']]


# Transform feature to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


# Split Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


# Ridge Regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)


# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)


# Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE: ", ridge_mse)

# Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print('Lasso Regression MSE: ', lasso_mse)


# Visualize Ridge vs Lasso predictions
plt.figure(figsize=(10,6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual Data', alpha=0.5)
plt.scatter(X_test[:, 0], ridge_predictions, color='green', label='Ridge Predictions', alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color='orange', label='Lasso Predictions', alpha=0.5)
plt.title('Ridge vs Lasso Regression')
plt.xlabel('Median Income in California (Transformed)')
plt.ylabel('Median House values in California')
plt.legend()
plt.show()
