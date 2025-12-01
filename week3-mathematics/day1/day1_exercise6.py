# create a block diagonal matrix using Numpy
# Block diagonal matrix: square matrix where the main diagonal consists of square submatrices, and all other off-diagonal elements are zero matrices
# A = [[A1, 0, 0], [0, A2, 0], [0, 0, A3]]
# where A1/A2/A3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# 0 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

import numpy as np
from scipy.linalg import block_diag

# defining individual blocks (matrices)
matrix1 = np.array([[1, 2], [3, 4]])

matrix2 = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])

matrix3 = np.array([[20]])

# create block diagonal matrix
block_diagonal_matrix = block_diag(matrix1, matrix2, matrix3)
print("Block Diagonal Matrix: \n", block_diagonal_matrix)

# Result:
# [[ 1  2  0  0  0  0]
#  [ 3  4  0  0  0  0]
#  [ 0  0  5  6  7  0]
#  [ 0  0  8  9 10  0]
#  [ 0  0 11 12 13  0]
#  [ 0  0  0  0  0 20]]