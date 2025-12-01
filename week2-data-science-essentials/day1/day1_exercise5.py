# generate a random array and find the minimum and maximum values

import numpy as np

# generates 3x4 matrix with random floating numbers, range [0,1)
random_array_float = np.random.rand(3,4)
print("Random Matrix Float:\n", random_array_float)

# generates 3x4 matrix with random integer numbers
# Syntax: np.random.randint(low, high, size)
random_array_int = np.random.randint(0, 10, size=(3,4)) # range [0-10)
print("Random Matrix Integer:\n", random_array_int)

# generates an array of specified shape (d0, d1, ...., dn) with random floating-point numbers drawn from standard normal distribution (mean 0, variance 1)
random_array_normal = np.random.randn(2,2)  # size 2x2
print("Random Matrix Normal Distribution:\n", random_array_normal)

# generates random sample from given 1-D array
# size for output shape, replace means with or without replacement, p for probabilities
choices = [1, 2, 3, 4, 5]
random_selection = np.random.choice(choices, size=3, replace=False)  # select 3 unique elements from 'choices'
print("Random Selection:\n",random_selection)

# using generator
rng = np.random.default_rng()  # create default random number generator range [0,1)
random_array_generator = rng.random((2, 3))  # generate a 2x3 matrix of random floats
print("Random Array Generator:\n", random_array_generator)