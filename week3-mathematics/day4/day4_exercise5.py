# Use Adam optimizer for a small and simple dataset

# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# sample dataset (linear regression)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2.5, 3.5, 4, 5])


# Adding bias term for theta0
X_b = np.c_[np.ones(len(X)), X]


# Define Cost Function
def compute_cost(theta):
  predictions = X_b.dot(theta)
  return (1/(2*len(y))) * np.sum((predictions - y)**2)


# Define Adam Optimizer Function
# beta1 and beta2: decay rates for first and second moments
# m_t: moving average of gradients (momentum)
# v_t: moving average of squared gradients (RMSProp-like scaling)
# m_hat and v_hat: bias-corrected estimates
# epsilon: prevents division by zero
# theta: updates each iteration using both momentum and scaling
def adam_optimizer(X, y, theta, learning_rate, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
  m = len(y)
  path = []
  
  # Initialize mpment vectors
  m_t = np.zeros_like(theta)  # first moment
  v_t = np.zeros_like(theta)  # second moment
  
  for t in range(1, iterations + 1):
    predictions = X.dot(theta)
    gradients = (1/m) * X.T.dot(predictions - y)
    
    # update biased first moment estimate
    m_t = beta1 * m_t + (1 - beta1) * gradients
    
    # update biased second raw moment estimate
    v_t = beta2 * v_t + (1 - beta2) * (gradients ** 2)
    
    # correct bias
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)
    
    # update paramentes
    theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    path.append(theta.copy())
    
  return np.array(path)


# Initialize and Run Adam Optimizer
theta_init = np.array([0.0, 0.0])
iterations = 1000
learning_rate = 0.1

# Run Adam
path_adam = adam_optimizer(X_b, y, theta_init.copy(), learning_rate, iterations)
print("Adam Optimizer: \n", path_adam)


# Plot Convergence
plt.figure(figsize=(10,6))
plt.plot(path_adam[:,1], label='theta1 (slope)')
plt.plot(path_adam[:,0], label='theta0 (intercept)')
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title('Adam Optimizer Convergence')
plt.legend()
plt.show()