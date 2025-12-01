# program to normalize an array (scale values between 0 and 1)

import numpy as np

def normalize_to_range(arr, desired_min, desired_max):
  array_min = np.min(arr)
  array_max = np.max(arr)
  
  # handle the case where all elements are the same(to avoid division by 0)
  if array_max == array_min:
    return np.full_like(arr, (desired_min + desired_max) / 2) # assign midpoint
  
  normalized_array = (arr - array_min) / (array_max - array_min) * (desired_max - desired_min) + desired_min
  return normalized_array

my_array = np.array([10, 20, 5, 30, 15])
normalized_array_0_1 = normalize_to_range(my_array, 0, 1)
print("Original Array:\n", my_array)
print("Normalized to [0, 1]:\n", normalized_array_0_1)

normalized_array_neg1_1 = normalize_to_range(my_array, -1, 1)
print("Normalized to [-1, 1]:\n", normalized_array_neg1_1)