import numpy as np

# creating matrix
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# matrix element-wise addition and subtraction
print("Addition: \n", A + B)
print("Subtraction: \n", B- A)

# scalar multiplication
C = 2 * A
print("Scalar multiplcation: \n", C)

# matrix multiplication
result = np.dot(A, B)
print("Matrix multiplication: \n", result)

# identity matrix
I = np.eye(3)
print("Identity matrix: \n", I)

# Zero matrix
Z = np.zeros((2, 3))
print("Zero matrix: \n", Z)

# Diagonal matrix
D = np.diag([1, 2, 3])  # giving values (1, 2, 3) to diagonal
print("Diagonal matrix: \n", D)

