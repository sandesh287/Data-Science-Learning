# # importing entire module
# import math
# print(math.sqrt(16))


# # importing specific function
# from math import sqrt
# print(sqrt(16))


# # importing using aliases
# import math as m
# print(m.sqrt(16))


# # function syntax
# def function_name(parameters):
#   return function_name


# # create a function with parameters and return value
# def add_numbers(a, b):
#   # c = a + b
#   return a + b

# result = add_numbers(5, 3)
# print("Sum: ", result)


# # local scope
# def greet():
#   message = "Hello World!"
#   print(message)
  
# greet()
# # print(message) # doesn't find message variable


# # global scope
# greeting = "Hi"

# def say_hello():
#   print(greeting + "from inside function")
  
# say_hello()
# print(greeting + "from outside function") # declared as global so no error