# Compute Eigenvalues and Eigenvectors

import numpy as np

# creating matrix
A = np.array([[4, -2], [1, 1]])

# eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigen Values: \n", eigenvalues)
print("Eigen Vectors: \n", eigenvectors)