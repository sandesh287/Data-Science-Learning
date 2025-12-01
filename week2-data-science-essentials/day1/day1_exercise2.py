# create 3x3 matrix and perform operations

import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Original Matrix:\n", matrix)

# Transpose of a matrix
transpose = matrix.T
print("Transpose:\n", transpose)


# element-wise operation
another_matrix = np.array([[9,8,7], [6,5,4], [3,2,1]])
print("Element-wise Addition:\n", matrix + another_matrix)
print("Element-wise Multiplication:\n", matrix * another_matrix)