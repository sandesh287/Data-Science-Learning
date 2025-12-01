# Regular expressions
import re  # importing regular expressions

text = "Contact me at 123-456-7890"
digits = re.findall(r"\d+", text)  # \d is for digits
digits_one = re.findall(r"\d", text)
print(digits_one) # displays: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
print(digits)  # displays: ['123', '456', '7890']

updated_text = re.sub(r"\d", "X", text)  # if given + after \d, it would replace whole 123 as X
print(updated_text)  # displays: Contact me at XXX-XXX-XXXX