import numpy as np

# creating arrays

# from list
arr = np.array([1,2,3,4])
print(arr)

# using build-in function
zeros = np.zeros((3,3))  # create 3x3 matrix with all 0
print(zeros)

ones = np.ones((2,4))  # create 2x4 matrix of ones
print(ones)

range_array = np.arange(1,10,2) # range from 1-10 with spacing of 2
print(range_array)  # [1 3 5 7 9]

linspace_array = np.linspace(0,1,5)  # returns evenly spaced numbers over interval of 5
print(linspace_array)  # prints: [0 0.25 0.5 0.75 1]


# manipulating array

arr = np.array([1,2,3,4,5,6])
reshaped = arr.reshape((3,2))  # forms matrix of 3x2
print(reshaped)

arr = np.array([1,2,3])
expanded = arr[:, np.newaxis]  # creates different dimension for each element
print(expanded) # prints: [[1]
                        #  [2]
                        #  [3]]
                        

# basic operations on array

a = np.array([1,2,3])
b = np.array([4,5,6])
print(a + b)  # displays: [5 7 9]
print(a * b)  # displays: [ 4 10 18]
print(a / b)  # displays: [0.25 0.4  0.5 ]

arr = np.array([4,16,25])
print(np.sqrt(arr))  # square root  [2 4 5]
print(np.sum(arr))  # sum of all numbers: 45
print(np.mean(arr))  # mean: 15
print(np.max(arr))  # max number among arr: 25


# indexing, slicing and reshaping array

arr = np.array([10,20,30,40,50,60])
print(arr[2])  # indexing:  30
print(arr[-1])  # negative indexing: 50

print(arr[1:4])  # include items of index 1,2,3 excluding 4: [20,30,40]
print(arr[:3])  # include items of index before 3, i.e. from start, but not 3: [10,20,30]
print(arr[3:])  # include items of index after 3 including 3 as well to the end: [40,50,60]

reshaped_array = arr.reshape(2,3)  # reshape the arr in 2x3 matrix form
print(reshaped_array)