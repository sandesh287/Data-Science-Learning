# Perform Singular Value Decomposition (SVD)

import numpy as np

# creating matrix
A = np.array([[2, 3, 4], [3, 1, 1], [-1, 3, 1]])

# SVD
U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)

# reconstruct the matrix to original one
Sigma = np.zeros((3, 3))
np.fill_diagonal(Sigma, S)
reconstructed = U @ Sigma @ Vt
print("Reconstructed Matrix:\n", reconstructed)