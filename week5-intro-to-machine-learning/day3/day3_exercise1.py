# Implement Polynomial Regression and Visualize the Fit
# Objective: To use polynomial regression to model the relationship between median income and median house values
# Steps: 
# 1. Load the California housing dataset
# 2. Select the feature median income and target variable median house value
# 3. Transform the feature into Polynomial features
# 4. Fit a polynomial regression model
# 5. Visualize the fitted curve and analyze the performance



# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load the california housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame


# Select feature (median income) and target (median house value)
X = df[['MedInc']]
y = df[['MedHouseVal']]


# Transform feature to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


# Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)


# Plot Actual vs Predicted values
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.5)
plt.scatter(X, y_pred, color='red', label='Predicted Curve', alpha=0.5)
plt.title('Polynomial Regression')
plt.xlabel('Median Income in California')
plt.ylabel('Median House values in California')
plt.legend()
plt.show()


# Evaluate model performance
mse = mean_squared_error(y, y_pred)
print('Mean Squared Error: ', mse)