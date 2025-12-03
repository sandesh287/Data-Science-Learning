# Implement Mini-Batch SGD and compare it with vanilla SGD in terms of convergence behaviour

# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# sample data (simple linear regression)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2.5, 3.5, 4, 5])


# adding bias term for theta0
# creates X_b as a 2D array with first column as 1(bias/intercept term) and second column as X
# if X = [1,2], then X_b = [[1,1], [1,2]]
X_b = np.c_[np.ones(len(X)), X]


# Define Cost Function
# theta: vector parameters [theta0, theta1]
# predictions = X_b.dot(theta): computes the linear model output
# We calculate Mean Squared Error(MSE) divided by 2 for simplicity in gradients
# compute_cost returns the scalar loss value
def compute_cost(theta):
  predictions = X_b.dot(theta)
  return (1/(2*len(y))) * np.sum((predictions - y)**2)


# Vanilla SGD Function
def vanilla_SGD(X, y, theta, learning_rate, iterations):
  path = []
  m = len(y)
  
  for _ in range(iterations):
    i = np.random.randint(m)  # pick a random sample
    xi = X[i:i+1]  # shape (1, n_features)
    yi = y[i]
    gradients = xi.T.dot(xi.dot(theta) - yi)  # computes gradients using single sample
    theta -= learning_rate * gradients  # updated theta
    path.append(theta.copy())  # saves theta in path to visualize the optimization path later
  
  return np.array(path)  # returns all theta value over iterations


# Mini-Batch SGD Function
# Mini-batch SGD is a compromise between full-batch gradient descent and vanilla SGD: it smooths updates and converges faster than vanilla SGD in practice.
# randomly selects batch_size samples per iteration
def mini_batch_SGD(X, y, theta, learning_rate, iterations, batch_size):
  path = []
  m = len(y)
  
  for _ in range(iterations):
    indices = np.random.choice(m, batch_size, replace=False)  # random batch
    X_batch = X[indices]
    y_batch = y[indices]
    
    gradients = (1/batch_size) * X_batch.T.dot(X_batch.dot(theta) - y_batch) # compute gradient
    theta -= learning_rate * gradients
    path.append(theta.copy())
    
  return np.array(path)


# Initialize parameters and Run both Algorithms
theta_init = np.array([0.0, 0.0])
learning_rate = 0.1
iterations = 50
batch_size = 2

# Run Vanilla SGD
path_vanilla_sgd = vanilla_SGD(X_b, y, theta_init.copy(), learning_rate, iterations)
print("Vanilla SGD: \n", path_vanilla_sgd)

# Run Mini-Batch SGD
path_mini_batch_sgd = mini_batch_SGD(X_b, y, theta_init.copy(), learning_rate, iterations, batch_size)
print("Mini-Batch SGD: \n", path_mini_batch_sgd)


# Visualize COnvergence
plt.figure(figsize=(10,6))
plt.plot(path_vanilla_sgd[:,1], label='Vanilla SGD', marker='o')
plt.plot(path_mini_batch_sgd[:,1], label='Mini-Batch SGD', marker='x')
plt.axhline(y=1, color='r', linestyle='--', label='True theta1')  # approximate true theta1
plt.xlabel('Iteration')
plt.ylabel('theta1 value')
plt.title('Comparison: Vanilla SGD vs Mini-Batch SGD')
plt.legend()
plt.show()