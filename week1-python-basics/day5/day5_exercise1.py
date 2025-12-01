# create a text cleaner

import re

def clean_text(text):
  # remove punctuation
  text = re.sub(r"[^\w\s]", "", text)
  # remove extra spaces
  text = " ".join(text.split())
  # convert to lowercase
  return text.lower()

input_text = "    Hello, World.!!! Welcome to Python, Programming....    "
cleaned_text = clean_text(input_text)
print("Original text: ", input_text)
print("Cleaned Text: ", cleaned_text)