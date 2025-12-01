# List
numbers = [1, 2, 3, 4]  # list of integers

fruits = ["apple", "banana", "mango"] # list of strings

mixed_list = [1, "apple", True] # mixed list

# accessing items inside list through index
print(numbers[2])
print(fruits[0])
print(mixed_list[1])

# negative indexing
print(fruits[-1]) # prints mango

# modifying list
# adding items
fruits.append("orange") # added orange to list at the end
fruits.insert(1, "grapes") # grapes inserted in index 1
print(fruits)

# remove items
fruits.remove("banana") # removing by name
del fruits[0] # removing by using index
fruits.pop() # remove last item
print(fruits)

# slicing list
sliced_fruits = fruits[1:3] # index 1, but before 3 i.e. 1,2
print(sliced_fruits) # prints items in index 1 and 2, but not 3


# Tuple
colors = ("red", "green", "blue") # 3 items tuple
single_item = ("glass",) # comma is necessary for single item tuple

print(colors[0]) # indexing
print(colors[-1]) # negative indexing


# Dictionaries
student = {"name": "Alice", "age": 25, "grade": "A"} # key:value
print(student) # prints everything inside student

print(student["name"]) # access value of name

student["subject"] = "Math"  # adds "subject": "Math" at last
student["age"] = 32 # updates age from 25 to 32

#removing items
del student["grade"] # removes grade

student.pop("subject") # need to mention what to pop, here pops subject

# Iteration of dictionary
for key, value in student.items():  # student.items() puts everything inside list
  print(key, value)
  

# Sets
numbers = {1, 2, 3, 4} # {} with items means set, {} with key-value pair means dictionary
empty_set = set() # create empty sets

# add items
numbers.add(5) # adds 5 at last
numbers.add(4) # nothing changes, as 4 is already present and set doesn't accept duplication
# as long as the items are not duplicate, we can add

# remove items
numbers.remove(2) # removes 2 from set

# union of set
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print(set1 | set2) # union of set1 and set2, prints {1, 2, 3, 4, 5}

# intersection of set
print(set1 & set2) # intersection of set1 and set2, prints {3}

# difference of set
print(set1 - set2) # prints {1, 2}