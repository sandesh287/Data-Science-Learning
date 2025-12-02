# Visualize the gradient descent process in quadratic function

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# define quadratic function and its gradient
def f(x):
  return x**2

def df(x):
  return 2*x

# define gradient descent function
def gradient_descent(start_x, learning_rate, iterations):
  x = start_x
  history = [x]
  
  for _ in range(iterations):
    grad = df(x)
    x -= learning_rate * grad
    history.append(x)
  return history

# parameters
start_x = 10
iterations = 30
learning_rates = [0.1, 0.2, 0.5]

# run gradient descent for different learning rates
histories = {}
for lr in learning_rates:
  histories[lr] = gradient_descent(start_x, lr, iterations)
  
# Plot
x_vals = np.linspace(-12, 12, 400)
plt.plot(x_vals, f(x_vals), label="f(x)=x²")

# Plot descent paths
for lr, hist in histories.items():
  plt.plot(hist, f(np.array(hist)), marker='o', label=f"LR = {lr}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent Visualization on f(x)=x²")
plt.legend()
plt.grid(True)
plt.show()