# Verify the property of eigenvalues: det(A - λI) = 0

# importing library
import numpy as np

# creating dataset / matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# eigenvalues and eigenvectors
eigenvalue, eigenvector = np.linalg.eig(A)
print("Eigenvalue: \n", eigenvalue)
print("Eigenvector: \n", eigenvector)

# For: det(A - λI) = 0 to verify, where λ is eigenvalue
# identity matrix and zero matrix
I = np.eye(A.shape[0])
print("Identity Matrix related to A: \n", I)
Z = np.zeros((3, 3))
print("Zero Matrix: \n", Z)
# dot product of eigenvalue and Identity matrix
dotLambdaIdentity = np.dot(eigenvalue, I)
print("Dot Product of Lambda and Identity Matrix: \n", dotLambdaIdentity)
result = (A - dotLambdaIdentity)
print("Subtraction of DotLambdaIdentity from Original Matrix: \n", result)
determinant = np.linalg.det(result)
print("Determinant of result: \n", determinant)

if np.allclose(determinant, Z):
  print("The property of Eigenvalues is verified.")
else:
  print("Failed!")