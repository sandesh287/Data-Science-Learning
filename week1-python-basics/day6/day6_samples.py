# open and read file
with open("sample.txt", "r") as file:
  content = file.read()
  print(content)
  
# open and write in file
with open("sample.txt", "w") as file:
  file.write("Hello World")
  file.writelines(["Alice", "Bob", "Cherry"]) # write in multiple lines
  
# file is automatically closed if we use "with" statement

# using exception handling
try:
  with open("sample.txt", "r") as file:
    content = file.read()
except FileNotFoundError:
  print("File Not Found!")