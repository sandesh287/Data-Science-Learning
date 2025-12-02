# Calculate determinant and inverse of a matrix

import numpy as np

# create matrix
A = np.array([[4, 2, 3], [4, 5, 6], [7, 8, 9]])

# determinant
determinant = np.linalg.det(A)
print("Determinant of A: \n", determinant)

# inverse
inverse = np.linalg.inv(A)
print("Inverse of A: \n", inverse)