# function to check if a number is even or odd and call it within another function

def odd_even(n):
  if n % 2 == 0:
    return f"{n} is an even number."
  else:
    return f"{n} is an odd number."
  
def odd_even_checker(n):
  result = odd_even(n)
  print(result)
  
odd_even_checker(10)