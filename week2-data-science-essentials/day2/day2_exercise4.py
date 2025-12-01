# program to generate a dataset of random floats and normalize the values between 0 and 1

import numpy as np

# generate a dataset of random floats
# using np.random.rand() to generate random values between 0 and 1 directly
# added 1, just to normalize the dataset in between 0 and 1
random_floats = np.random.rand(2,3) + 1
print("Original dataset:\n", random_floats)

# normalize the values between 0 and 1 (if not already in that range)
# only necessary for datasets not already in the range [0, 1).
min_value = np.min(random_floats)
max_value = np.max(random_floats)

if min_value == max_value:
  normalized_dataset = np.zeros_like(random_floats)
else:
  normalized_dataset = (random_floats - min_value) / (max_value - min_value)
  
print("Normalized dataset (from uniform distribution):\n", normalized_dataset)

# verify the min and max of normalized dataset
print(f"Min value of normalized dataset: {np.min(normalized_dataset)}")
print(f"Max value of normalized dataset: {np.max(normalized_dataset)}")
