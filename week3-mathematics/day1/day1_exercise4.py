# Compute the determinant and inverse of a 2x2 matrix using NumPy

import numpy as np

# creating 2x2 matrix
original_matrix = np.array([[1, 2], [3, 4]])

# Determinant of matrix
# Using np.linalg.det() function
determinant_matrix = np.linalg.det(original_matrix)
print("Determinant matrix of A: \n", determinant_matrix)

# Inverse of matrix
inverse_matrix = np.linalg.inv(original_matrix)
print("Inverse of A: \n", inverse_matrix)

# Verify the inverse is correect or not
# multiply original matrix with its calculated inverse matrix, result should be identity matrix
identity_check = np.dot(original_matrix, inverse_matrix)
print("Verifying Inverse: \n", np.round(identity_check, 10))

# Before using np.round(), the answer was showing 8.8817842e-16 which is nearly equal to zero. It appears because of floating-point precision limitations in computers.
# Using np.round( , 10), rounded the value to tens and the result is [[1. 0.]
#                                                                     [0. 1.]]
