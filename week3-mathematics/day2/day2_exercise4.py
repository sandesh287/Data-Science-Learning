# Compute eigenvalues and eigenvectors for larger matrices

import numpy as np

# dataset (creating matrix)
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

# eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues: \n", eigenvalues)
print("Eigenvectors: \n", eigenvectors)