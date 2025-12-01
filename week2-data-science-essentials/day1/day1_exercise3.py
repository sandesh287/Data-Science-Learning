# create 4x4 matrix and calculate sum of its rows and columns

import numpy as np

matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])

print("Original matrix:\n", matrix)

# sum of all elements
total_sum = np.sum(matrix)
print(total_sum)  # prints: 136

# sum of each rows
row_sums = np.sum(matrix, axis=1)
print(row_sums)  # prints: [10 26 42 58]

# sum of each columns
column_sums = np.sum(matrix, axis=0)
print(column_sums)  # prints: [28 32 36 40]