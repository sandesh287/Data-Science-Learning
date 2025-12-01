# Generate and Filter a random dataset

import numpy as np

# generate random dataset
dataset = np.random.randint(1, 51, size=(5,5))
print("Original Dataset:\n", dataset)

# filter values > 25 and replace with 0
dataset[dataset > 25] = 0
print("Modified dataset:\n", dataset)

# calculate summary stats
print("Sum: ", np.sum(dataset))
print("Mean: ", np.mean(dataset))
print("Standard Deviation: ", np.std(dataset))