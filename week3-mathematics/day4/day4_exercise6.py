# Use Adam optimizer for a more complex and large dataset(many samples, multiple features)
# synthetic multi-dimensional regression dataset created via scikit-learn

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 1. Creating a synthetic regression dataset
X, y = make_regression(
  n_samples=1000,  # 1000 data points
  n_features=20,  # 20 features (more realistic dimensionality)
  noise=20.0,  # add noise for realistic complexity
  random_state=42
)


# 2. Feature scaling (important for optimization convergence)
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)


# 3. Add bias term (intercept)
X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]


# 4. Split into train and test (to check performance)
X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)


# 5. Define cost function (MSE)
def compute_cost(X, y, theta):
  m = len(y)
  predictions = X.dot(theta)
  return (1 / (2*m)) * np.sum((predictions - y)**2)


# 6. Adam Optimizer Function implementation
def adam_optimizer(X, y, theta, lr=0.01, epochs=500, beta1=0.9, beta2=0.999, epsilon=1e-8):
  m = len(y)
  m_t = np.zeros_like(theta)
  v_t = np.zeros_like(theta)
  cost_history = []
  for t in range(1, epochs + 1):
    predictions = X.dot(theta)
    gradients = (1/m) * X.T.dot(predictions - y)
    
    # Update biased first and second moment estimates
    m_t = beta1 * m_t + (1 - beta1) * gradients
    v_t = beta2 * v_t + (1 - beta2) * (gradients ** 2)
    
    # Bias‑corrected estimates
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)
    
    # Parameter update
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    cost_history.append(compute_cost(X, y, theta))
  return theta, np.array(cost_history)


# 7. Run Adam on training data
n_features = X_train.shape[1]
theta_init = np.zeros(n_features)  # initialize all parameters to zero
theta_adam, cost_hist = adam_optimizer(X_train, y_train, theta_init, lr=0.05, epochs=500)

print("Final Training Cost (Adam): ", cost_hist[-1])


# 8. Evaluate on test set
test_cost = compute_cost(X_test, y_test, theta_adam)
print("Test Cost (Adam): ", test_cost)


# Plot cost over epochs (convergence curve)
plt.figure(figsize=(8,5))
plt.plot(cost_hist, label='Adam Training Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE/2)')
plt.title('Convergence of Adam on Complex Dataset')
plt.legend()
plt.show()





# 1. What this demonstrates

# We generated 1000 samples × 20 features — much more realistic than a 2‑ or 3‑sample toy example.

# We scaled features (standardization), which is often necessary in real-world ML before applying gradient‑based optimizers.

# The model (a linear regression) is trained using Adam; we track and plot the cost over iterations.

# We finally evaluate on a held-out test set to check generalization (not just overfitting training data).

# The convergence curve will show whether Adam smoothly converged; you can visually inspect speed and stability.

# 2. How you can extend / experiment further

# Try different learning rates (lr), batch sizes (convert to mini‑batch Adam), or number of iterations to see how they affect convergence.

# Use a non‑linear dataset (e.g. generated with polynomial features or non-linear target) and then use a more complex model (e.g. a small neural network) — Adam is especially useful there.

# Compare with vanilla SGD or mini‑batch SGD on the same dataset to see performance differences in cost curve, convergence speed, and generalization error.

# Add regularization (Ridge/Lasso) or feature engineering / normalization to make the task more realistic.