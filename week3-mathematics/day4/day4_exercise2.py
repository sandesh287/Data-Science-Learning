# Implement Stochastic Gradient Descent (SGD) for a Linear Model
# objective: implement SGD to minimize a linear regression loss function

# importing library
import sympy as sp
import numpy as np

# Generate synthetic data
np.random.seed(42)  # .seed() make sure every time we run randomizer, we get same random data
X = 2 * np.random.rand(100, 1)  # this is my feature
y = 4 + 3 * X + np.random.randn(100, 1)  # this will give me target with some noise value

# Add biased term to X
X_b = np.c_[np.ones((100, 1)), X]

# SGD implementation
def stochastic_gradient_descent(X, y, theta, learning_rate, n_epochs):
  m = len(y)
  for epoch in range(n_epochs):
    for i in range(m):
      random_index = np.random.randint(m)
      xi = X[random_index:random_index+1]
      yi = y[random_index:random_index+1]
      gradients = 2 * xi.T @ (xi @ theta - yi)
      theta -= learning_rate * gradients
  return theta

# initialized parameters
theta = np.random.randn(2, 1)
learning_rate = 0.01
n_epochs = 50

# Perform SGD
theta_optimized = stochastic_gradient_descent(X_b, y, theta, learning_rate, n_epochs)
print("Optimized Parameters: ", theta_optimized)