# calculate the factorial of a number using while loop

def factorial(n):
  if n < 0:
    return "Factorial of negative number does not exist"
  elif n == 0:
    return 1
  else:
    factorial_result = 1
    counter = 1
    while counter <= n:
      factorial_result *= counter
      counter += 1
    return factorial_result

num = int(input("Enter a positive number: "))

print(f"Factorial of {num} is: {factorial(num)}")