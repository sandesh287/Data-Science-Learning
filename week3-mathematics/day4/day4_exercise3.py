# Visualize the loss function's surface and the SGD optimization path

# We will optimize a simple linear regression: hθ​(x)=θ0​+θ1​x
# Loss Function: J(θ0​,θ1​)=2m1​∑(hθ​(x)−y)2

# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# sample data
# The model we are going to fit is y ≈ theta0 + theta1 * x
X = np.array([1, 2, 3])
y = np.array([2, 2.5, 3.5])


# add bias term
# np.ones(len(X)) create a 1D array of ones with length 3: [1, 1, 1]. This is bias column
# np.c_[ ... ] concatenates the columns. So X_b becomes a 2D array with shape 3x2.
# [[1., 1.],
#  [1., 2.],
#  [1., 3.]]
# First column is the bias(for theta0), second column is the original feature(for theta1)
# X_b is the design matrix for linear regression with an intercept
X_b = np.c_[np.ones(len(X)), X]


# Cost Function
# This function computes the mean squared error cost for a parameter vector theta(length 2: [theta0, theta1])
# predictions = X_b.dot(theta): matrix vector product. If theta is shape (2,), predictions is shape (3,) then: predictions[i] = theta0 * 1 + theta1 * X[i]
# (predictions - y): element-wise residuals (shape (3,))
# ((predictions - y)**2): squared residuals, element-wise
# np.sum(...): sum of squared erros
# (1/(2*len(y))) * ... : scales the sum by (1/2m) where m is number of examples (3 here). The factor 1/2 is conventional for linear regression because it cancels the 2 when differentiating the square term
def compute_cost(theta):
  predictions = X_b.dot(theta)
  return (1/(2*len(y))) * np.sum((predictions - y)**2)


# SGD Function
# X is expected to be design matrix(here X_b of shape(3,2)), y is target vector(shape (3,)), theta is initial parameters(shape(2,)), learning_rate is scalar, iterations is number of SGD steps to run
# path=[]: list to store parameter values at every step(for plotting optimized trajectory)
# m=len(y): number of training examples(3)
# for _ in range(iterations): loop for iterations steps
# i=np.random.randint(m): pick random integer in [0, m-1], selects random training example
# xi=X[i:i+1]: slice returns a 2D row matrix(shape (1,2)) instead of 1D, which keeps matrix-vector multiplication rules consistent
# yi=y[i]: scalar target for the selected example
# gradients = xi.T.dot(xi.dot(theta) - yi):
# xi.dot(theta): computes the prediction for single sample; shape (1,)
# (xi.dot(theta) - yi): scalar residual for that sample
# xi.T: has shape (2,1)
# xi.T.dot( ... ): yields shape (2,) effectively the gradient of the squared error for this single sample w.r.t theta
# Note: This gradient is per-sample gradient of (1/2)*(prediction - yi)^2, because its not averaged across the dataset, the learning rate must be chosen accordingly
# theta -= learning_rate * gradients: parameter update step
# path.append(theta.copy()): append a copy of current theta to preserve its value at this step
# return np.array(path): convert the list of theta vectors into 2D NumPy array of shape (iterations, 2). Each row is [theta0, theta1] after this step
def SGD(X, y, theta, learning_rate, iterations):
  path = []
  m = len(y)
  
  for _ in range(iterations):
    i = np.random.randint(m)  # random sample each step
    xi = X[i:i+1]
    yi = y[i]
    gradients = xi.T.dot(xi.dot(theta) - yi)
    theta -= learning_rate * gradients
    path.append(theta.copy())
    
  return np.array(path)


# Generate loss surface (theta0, theta1 grid)
# np.linespace(-1, 3, 100): create 100 evenly spaced values from -1 to 3 for theta0
# np.linspace(-2, 2, 100): create 100 evenly spaced values from -2 to 2 for theta1
# np.meshgrid(t0_vals, t1_vals): creates 2 coordinates matrices T0 and T1, both shape (100,100), where each (j,i) pair corresponds to a point (theta0, theta1) on the grid:
# T0[j,i] == t0_vals[i]
# T1[j,i] == t1_vals[j]
# J_vals = np.zeros_like(T0): initialize a (100,100) array of zeros to hold cost values for each (theta0, theta1) grid point
t0_vals = np.linspace(-1, 3, 100)
t1_vals = np.linspace(-2, 2, 100)

T0, T1 = np.meshgrid(t0_vals, t1_vals)
J_vals = np.zeros_like(T0)


# Nested loops iterate over grid and compute cost for each combination:
# looping i across t0_vals(x direction), and j across t1_vals(y direction)
# t = np.array([T0[j, i], T1[j, i]]): constructs a parameter vector [theta0, theta1]. Note the ordering T0[j,i] then T1[j,i] matches theta convention used everywhere
# J_vals[j, i] = compute_cost(t): computes cost and stores it at corresponding grid location. The indexing [j,i] keeps the orientation consistent with meshgrid output for plotting
# Note: This double loop is fine for small grids, but for larger grids you might vectorize the cost computation for speed
for i in range(len(t0_vals)):
  for j in range(len(t1_vals)):
    t = np.array([T0[j, i], T1[j, i]])
    J_vals[j, i] = compute_cost(t)
    

# Initialize values and Run SGD
# theta_init: starting parameters(both zero)
# iterations=80: number of SGD updated to perform
# path=SGD(...): run the SGD routine and collect the optimization path(shape (80,2))
theta_init = np.array([0.0, 0.0])
learning_rate = 0.05
iterations = 80
path = SGD(X_b, y, theta_init, learning_rate, iterations)
print("Optimized Theta: ", path)


# Plot
# plt.figure: create a new figure(here sized 12x8 inches)
# plt.contour: draws contour lines of loss surface
# T0 and T1: coordinate grids, J_vals: function values, 50: draw 50 contour levels
# cmap='viridis': sets the colormap for contour lines/levels
# path[:,0]: sequence of theta0 values over iterations
# path[:,1]: sequence of theta1 values
# 'ro-': draws red circles connected by lines(r:red, o:circle, -:line)
# label: used for legend
# plt.title: sets title
# plt.xlabel and plt.ylabel: axis labels
# plt.legend(): draw legend
# plt.show(): display the plot
plt.figure(figsize=(12, 8))
plt.contour(T0, T1, J_vals, 50, cmap='viridis')
plt.plot(path[:,0], path[:,1], 'ro-', markersize=3, label="SGD Path")
plt.title("SGD Path on Loss Surface")
plt.xlabel("theta0")
plt.ylabel('theta1')
plt.legend()
plt.show()




# Extra notes, gotchas and suggestions

# 1. Randomness / reproducibility

# Because SGD picks random examples, each run produces a different path. For reproducible results, set np.random.seed(0) (or another integer) before calling SGD.

# 2. Gradient normalization

# Your gradients = xi.T.dot(xi.dot(theta) - yi) is the per-sample gradient of the squared error (no averaging). That’s OK for SGD, but learning rate must be tuned accordingly. Some versions divide by m or by 1 (single sample), or scale by 1 before updating.

# 3. Cost surface orientation

# In the grid loops you used J_vals[j,i] and t = [T0[j,i], T1[j,i]]. That ordering is correct given how meshgrid was constructed and how plt.contour expects T0/T1 shapes.

# 4. Using compute_cost while taking SGD steps

# If you want to track the loss along the path, compute compute_cost(theta) at each step and store it. That allows plotting loss vs iteration.

# 5. Vectorization tip

# Building J_vals by double loop is fine for 100×100, but a vectorized approach is faster for bigger grids.

# 6. Learning rate choices

# If lr is too large, SGD diverges. If too small, it converges slowly. Try a few values (0.001, 0.01, 0.05, 0.1) and inspect the path.

# 7. Plot aesthetics

# plt.plot(path[:,0], path[:,1], 'ro-') draws the optimization path on top of contours. If the path is noisy it may look jumpy (that’s expected for SGD).

# 8. Potential numerical issues

# For small problems like this there won’t be overflow, but for larger datasets or huge learning rates you can see overflow or exploding theta (as you observed earlier with high lr).