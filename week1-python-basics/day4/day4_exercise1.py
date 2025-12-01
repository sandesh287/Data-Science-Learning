# manipulate data in a dictionary
person = {"name": "Alice", "age": 25, "grade": "A"}
print(person)

# add new key-value pair
person["address"] = "123 Main St"

# update age
person["age"] = 32

# remove grade
if "grade" in person:
  del person["grade"]
  
print(person)