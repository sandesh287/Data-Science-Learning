# module for string operations imported by exercise 4

def reverse_string(input_string):
  return input_string[::-1]
  
def count_vowels(input_string):
  vowels = "aeiouAEIOU"
  vowel_count = 0
  
  for char in input_string:
    if char in vowels:
      vowel_count += 1
  
  return vowel_count

def palindrome_checker(input_string):
  if input_string == input_string[::-1]:
    return f"{input_string} is palindrome."
  else:
    return f"{input_string} is not palindrome."
  
