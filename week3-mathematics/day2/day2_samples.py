# Advanced Linear Algebra Concepts

import numpy as np

# creating matrix
A = np.array([[2, 3], [1, 4]])

# Determinant of a matrix
determinant = np.linalg.det(A)
print("Determinant: ", determinant)

# Inverse of matrix
inverse = np.linalg.inv(A)
print("Inverse of A: \n", inverse)

# Eigenvalues and Eigenvectors
eigenValues, eigenVectors = np.linalg.eig(A)
print("Eigenvalues: \n", eigenValues)
print("Eigenvectors: \n", eigenVectors)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)