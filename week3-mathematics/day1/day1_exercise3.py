# Explore Special Matrices

import numpy as np

# creating 3x3 matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Identity 3x3 matrix
I = np.eye(3)
print("Identity Matrix: \n", I)
print("Matrix and Identity matrix multiplication (A x I): \n", np.dot(A, I))

# Diagonal matrix
D = np.diag([1, 7, 9])
print("Diagonal Matrix: \n", D)

# Zero matrix
Z = np.zeros([3, 3])
print("Zero Matrix: \n", Z)