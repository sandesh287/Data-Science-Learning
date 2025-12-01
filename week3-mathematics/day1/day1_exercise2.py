# Implement Matrix-Vector Multiplication
# First we'll create 3x3 matrix and a 3D vector, then perform matrix-vector multiplication

import numpy as np

# creat 3x3 matrix and 3D vector
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([1, 0, -1])

# Matrix-Vector multiplication
result = np.dot(M, v)
print("Matrix-Vector Multiplication: \n", result)