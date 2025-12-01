# check if a number is prime
num = int(input("Enter a number: "))

if num > 1:
  end_point = int(num**0.5) + 1  # num**0.5 means square root of the number
  # print(end_point)
  for i in range(2, end_point):
    if num % i == 0:
      print(f"{num} is not a prime number")
      break
  else:
    print(f"{num} is a prime number")
else:
  print(f"{num} is not a prime number")