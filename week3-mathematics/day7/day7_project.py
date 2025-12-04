# Task 1: Implement the Mathematical Formula for Linear Regression
# Task 2: Use Gradient Descent to Optimize the Model Parameters
# Task 3: Calculate Evaluation Metrics


# Importing libraies
import numpy as np


# Generating Synthetic Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


# Adding bias term to feature matrix
X_b = np.c_[np.ones((100, 1)), X]


# Initializing parameters
theta = np.random.randn(2, 1)
learning_rate = 0.1
iterations = 1000


# Task 1: Implement the Mathematical Formula for Linear Regression

# predict function
def predict(X, theta):
  return np.dot(X, theta)


# Task 2: Use Gradient Descent to Optimize the Model Parameters
# Objective: To implement Gradient Descent to minimize the loss function

# gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
  m = len(y)
  for _ in range(iterations):
    gradients = (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
    theta -= learning_rate * gradients
  return theta


# Task 3: Calculate Evaluation Metrics
# Objective: To implement functions to calulate MSE and R-squared

# mean squared error function
def mean_squared_error(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

# R squared function
def r_squared(y_true, y_pred):
  ss_res = np.sum((y_true - y_pred)**2)
  ss_tot = np.sum((y_true - np.mean(y_true))**2)
  return 1 - (ss_res / ss_tot)


# Perform gradient descent
theta_optimized = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Predictions and Evaluations
y_pred = predict(X_b, theta_optimized)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)

print("Optimized Parameters (theta): ", theta_optimized)
print('MSE: ', mse)
print('R2: ', r2)