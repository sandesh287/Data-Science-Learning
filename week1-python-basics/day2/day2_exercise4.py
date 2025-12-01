# find the largest number in a list using for loop

# numbers = [1, 2, 3, 4, 5, 6]
user_input_string = input("Enter list of numbers separated by space: ")
numbers = list(map(int, user_input_string.split()))

if not numbers:
  print("The list is empty")
else:
  largest_number = numbers[0]
  for num in numbers:
    if num > largest_number:
      largest_number = num
  print(f"{largest_number} is the largest number.")
  