# reverse a list and remove duplicates using a set

my_list = [1, 2, 10, 3, 4, 5, 6, 7, 1, 4, 10]

reverse_list = my_list[::-1]

print(reverse_list)

# converting list into set simple way for removing duplicate elements
unique_list = list(set(my_list))
print(unique_list)

# using loop removing duplicate elements(preserves order)
no_duplicates = []
for item in my_list:
  if item not in no_duplicates:
    no_duplicates.append(item)
print(no_duplicates)