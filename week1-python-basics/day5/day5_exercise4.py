# find and replace all email address in a text using regex(regular expression)

import re

input_text = "Hello from sandesh@gmail.com to sanskriti@gmail.com about the meeting."

email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# extract email from text
email = re.findall(email_pattern, input_text)
print("Extracted email address: ", email)

# replace email with ***
replaced_email = re.sub(email_pattern, "*", input_text)
print("Replaced email: ", replaced_email)