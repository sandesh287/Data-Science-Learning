# create a 3D random array and compute statistics along specific axes

import numpy as np

array_3d = np.random.randint(1, 11, size=(3, 3, 3))
print("Original 3D array:\n",array_3d)

# Sum along axis 0 (across 1st dimension/plane)
# result: d11 + e11 + f11 = a11, similarly...
sum_axis_0 = np.sum(array_3d, axis=0)
print("Sum along axis 0:\n", sum_axis_0)
print("Shape of sum along axis 0: ", sum_axis_0.shape)

# Sum along axis 1 (across the 2nd dimension/rows within plane)
# d11 + d21 + d31 = a11, d12 + d22 + d32 = a12
sum_axis_1 = np.sum(array_3d, axis=1)
print("Sum along axis 1:\n", sum_axis_1)
print("Shape of sum along axis 1: ", sum_axis_1.shape)

# Sum along axis 2 (across the 3rd dimension/columns within plane)
# d11 + d12 + d13 = g11, d21 + d22 + d23 = g12, similarly
sum_axis_2 = np.sum(array_3d, axis=2)
print("Sum along axis 2:\n", sum_axis_2)
print("Shape of sum along axis 2: ", sum_axis_2.shape)