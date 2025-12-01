# create a module for string operations, including functions to reverse a string, count vowels, and check for palindromes. Import it into a script and test the functions.

# importing string_operations module
import day3.day3_string_operations as str

userInput = input("Enter a string value: ")

print(f"The reverse of {userInput} is {str.reverse_string(userInput)}")
print(f"The number of vowels in {userInput} is {str.count_vowels(userInput)}")
print(str.palindrome_checker(userInput))