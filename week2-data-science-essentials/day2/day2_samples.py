import numpy as np

# array and scalar broadcasting
arr = np.array([1, 2, 3])
print(arr + 10)  # adds 10 to each element

# array and vector broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([1, 0, 1])
print(matrix + vector) # adds vector to matrix and prints: [[2 2 4]
                                                          #  [5 5 7]]
                                                        

# Aggregation Functions
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Sum: ", np.sum(arr))
print("Mean: ", np.mean(arr))
print("Max: ", np.max(arr))
print("Min: ", np.min(arr))
print("Standard Deviation: ", np.std(arr))
print("Sum along rows: ", np.sum(arr, axis=1))
print("Sum along columns: ", np.sum(arr, axis=0))


# Boolean Indexing and Filtering
arr = np.array([1,2,3,4,5,6])

# Filter all the even numbers from array
evens = arr[arr % 2 == 0]
print("Evens: ", evens)

# modifying array based on particular conditions
# if value inside array are greater than 3, it changes into 0
arr[arr > 3] = 0
print("Modified Array: ", arr)


# Random Number Generation
# generating floating number range: [0,1)
random_array = np.random.rand(3, 3)
print("Random Array:\n", random_array)

# generating integers range [0, 10), size of matrix: (2, 3)
random_integers = np.random.randint(0, 10, size=(2, 3))
print("Random Integers:\n", random_integers)


# setting random seed
np.random.seed(42)

random_number = np.random.rand(3,3)
print("Random number array:\n", random_number)