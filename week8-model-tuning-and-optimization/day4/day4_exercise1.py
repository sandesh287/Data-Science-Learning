# Regularization Techniques for Model Optimization
# Apply Lasso and Ridge regularization on a linear regression model, compare performance, and analyze the effects on coefficients
# Dataset: California Housing Dataset



# libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


# Load dataset
california = fetch_california_housing()


# Features and Target
X, y = california.data, california.target
feature_names = california.feature_names


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display dataset info
print('Feature Names: \n', feature_names)
print('\nSample Data: \n', pd.DataFrame(X, columns=feature_names).head())



# Train Linear Regression Model without Regularization
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# Predict and evaluate
y_pred = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)

print(f'\n\nLinear Regression MSE (No Regularization): {mse_lr}')
print('Coefficients: \n', lr_model.coef_)



# Apply Ridge Regularization

# Train Ridge Regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)


# Predict and evaluate
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f'\n\nRidge Regression MSE: {mse_ridge}')
print('Coefficients: \n', ridge_model.coef_)



# Apply Lasso Regularization

# Train Lasso Regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)


# Predict and evaluate
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f'\n\nLasso Regression MSE: {mse_lasso}')
print('Coefficients: \n', lasso_model.coef_)