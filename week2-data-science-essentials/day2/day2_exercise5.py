# implement conditional replacement to create a binary mask for values above a threshold

import numpy as np

# create numpy array
data_array = np.array([10, 25, 5, 40, 15, 30, 8])

# define threshold
threshold = 20

# create boolean binary mask
# comparison operator to create a boolean array where True indicates value above threshold and False indicates value below or equal to threshold
boolean_binary_mask = data_array > threshold
print("Binary Mask: ", boolean_binary_mask)

# create a numerical binary mask (1s and 0s)
numerical_binary_mask = boolean_binary_mask.astype(int)
print("numerical Binary Mask (using astype(int)): ", numerical_binary_mask)

# create a numerical binary mask using np.where()
numerical_binary_mask_where = np.where(boolean_binary_mask, 1, 0)
print("Numerical Binary Mask (using np.where()): ", numerical_binary_mask_where)