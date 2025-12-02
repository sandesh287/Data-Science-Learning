# Use SVD to reduce the dimensionality of a dataset

# importing library
import numpy as np

# dataset
A = np.array([
  [1, 2, 3, 4, 5],
  [6, 7, 8, 9, 10],
  [11, 12, 13, 14, 15],
  [16, 17, 18, 19, 20]
])

# perform SVD
U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)

# choosing number of component
# deciding desired number of dimensions you want to reduce your data to.
k = 2  # reducing to 2 dimensions

# reconstruct the reduced dimensional data
# using formula: Xreduced = Uk.Sk , where Uk is first columns of U and Sk is diagonal matrix formed from the first k singular values
# select the first k columns of U and first k singular values
U_k = U[:, :k]
S_k = np.diag(S[:k])  # creating a diagonal matrix from singular values
# reconstruct the reduced data (projected onto the new basis)
data_reduced = U_k @ S_k
print("Reduced Matrix: \n", data_reduced)