# list comprehensions
# avoid multiple lines of code and write in one line

# [expression for item in iterable if condition]  (syntax)

# creating a list of squares
squares = [x**2 for x in range(10)]
print(squares)

# filter even numbers
evens = [x for x in range(10) if x % 2 == 0]
print(evens)


# Lambda function
# lambda arguments: expression  (syntax)

add = lambda x, y: x + y
print(add(3,5))

# the above line is similar to: def add(x, y): return x + y


# common methods
# map()
list_of_numbers = [1, 2, 3, 4]
square_list = map(lambda x: x**2, list_of_numbers)
print(list(square_list))

# filter()
evenList = filter(lambda x: x % 2 == 0, list_of_numbers)
print(list(evenList))

# reduce()
from functools import reduce
product = reduce(lambda x, y: x * y, list_of_numbers)
print(product)


# Python's os and sys module
# # os module
# import os
# print(os.getcwd())  # get current directory
# os.mkdir("test_dir") # make directory
# os.remove("file.txt") # remove file

# sys module
import sys
print(sys.argv) # gives command line argument
print(sys.version) # version of python