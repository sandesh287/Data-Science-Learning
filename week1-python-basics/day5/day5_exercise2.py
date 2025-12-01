# check if a string is a palindrome

def is_palindrome(text):
  text = "".join(char.lower() for char in text if char.isalnum()) # remove all alphanumeric characters, spaces, puntuations and convert to lowercase at once
  reverse_text = text[::-1]  #reverse the text
  return text == reverse_text  # checks if text is equal to reverse text

input_text = input("Enter a string: ")
if is_palindrome(input_text):
  print(f'"{input_text}" is a palindrome.')
else:
  print(f'"{input_text}" is not a palindrome.')