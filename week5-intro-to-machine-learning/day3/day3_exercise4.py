# Use Multiple Features:
  # Include more features (eg. HouseAge, AveRooms) and observe the impact on model performance
# Multi-Feature RIDGE and LASSO Regression
# Features: (HouseAge, AveRooms), Target: Meadian House Value
# Polynomial(degree=2) + Ridge + Lasso


# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load the california housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame


# Select feature (HouseAge, AveRooms) and target (median house value)
X = df[['HouseAge', 'AveRooms']]
y = df[['MedHouseVal']]


# Transform feature to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print('Original feature shape: ', X.shape)
print('Polynomial feature shape: ', X_poly.shape)


# Scale Features
scalar = StandardScaler()
X_poly_scaled = scalar.fit_transform(X_poly)


# Split Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)


# Linear Regression (Baseline)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_predictions)


# Ridge Regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_predictions)


# Lasso Regression
lasso_model = Lasso(alpha=0.001)  # smaller alpha so model doesn't kill all features
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_predictions)


# Print Results
print('\n--- Model Performance (MSE) ---')
print(f"Linear Regression MSE: {linear_mse}")
print(f"Ridge Regression MSE: {ridge_mse}")
print(f'Lasso Regression MSE: {lasso_mse}')

print('\n--- Number of polynomial features used ---')
print('Total Features: ', X_poly.shape[1])

print('\n--- Coefficients ---')
print(f'Linear Regression: {linear_model.coef_}')
print(f'Ridge Regression: {ridge_model.coef_}')
print(f'Lasso Regression: {lasso_model.coef_}')


# Visualize Actual vs Predictions
plt.figure(figsize=(10,6))

plt.scatter(y_test, linear_predictions, label='Linear', alpha=0.5)
plt.scatter(y_test, ridge_predictions, label='Ridge', alpha=0.5)
plt.scatter(y_test, lasso_predictions, label='Lasso', alpha=0.5)

plt.title('Polynomial Regression (Linear vs Ridge vs Lasso)')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted House Value')
plt.legend()
plt.grid(True)
plt.show()
