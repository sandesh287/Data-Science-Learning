# Implement gradient descent with multiple learning rates and compare convergence speeds

# importing library
import numpy as np
import matplotlib.pyplot as plt

# define gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
  m = len(y)
  cost_history = []
  
  for _ in range(iterations):
    predictions = np.dot(X, theta)
    errors = predictions - y
    gradients = (1/m) * np.dot(X.T, errors)
    theta -= learning_rate * gradients
    
    # compute cost
    cost = (1/(2*m)) * np.sum(errors**2)
    cost_history.append(cost)
    
  return theta, cost_history

# sample data
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 2.5, 3.5])

theta_init = np.array([0.1, 0.1])

learning_rate_1 = 0.1
learning_rate_2 = 0.3
iterations = 1000

# perform gradient descent separately with copies of initial data
optimized_theta_1, cost1 = gradient_descent(X, y, theta_init.copy(), learning_rate_1, iterations)
optimized_theta_2, cost2 = gradient_descent(X, y, theta_init.copy(), learning_rate_2, iterations)

print("Theta with learning rate = 0.1: ", optimized_theta_1)
print("Theta with learning rate = 0.3: ", optimized_theta_2)

# plot the convergence
plt.plot(cost1, label='lr = 0.1')
plt.plot(cost2, label='lr = 0.3')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence for Different Learning Rates")
plt.legend()
plt.show()