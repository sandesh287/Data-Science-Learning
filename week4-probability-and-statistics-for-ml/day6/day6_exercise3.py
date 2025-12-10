# Fit a multiple linear regression model with multiple independent variables
# Predict y using two independent variable x1 and x2


# libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate data
np.random.seed(42)

n = 100
x1 = np.random.rand(n, 1) * 10  # feature 1
x2 = np.random.rand(n, 1) * 5  # feature 2

# True relationship
y = 4*x1 + 2*x2 + np.random.randn(n, 1) * 3

# combine features
X = np.hstack((x1, x2))


# Fit Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)


# Get Coefficients
slope_x1, slope_x2 = model.coef_[0]
intercept = model.intercept_[0]
r_squared = model.score(X, y)

print("Intercept:", intercept)
print("Slope for x1:", slope_x1)
print("Slope for x2:", slope_x2)
print("R-squared:", r_squared)


# Create prediction grid
x1_grid, x2_grid = np.meshgrid(
  np.linspace(x1.min(), x1.max(), 20),
  np.linspace(x2.min(), x2.max(), 20)
)

X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
y_grid = model.predict(X_grid).reshape(x1_grid.shape)


# Visualize 3D plot
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

# scatter plot (actual data)
ax.scatter(x1, x2, y, color='blue', alpha=0.6, label='Actual Data')

# Regression plane
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Multiple Linear Regression: 3D Visualization')

plt.show()


# Blue dots: actual data points
# Plane: predicted values from the regression model
# Slope along x1 direction: effect of x1 on y (holding x2 constant)
# Slope along x2 direction: effect of x2 on y (holding x1 constant)