# program that counts the number of occurences of a specific word in a text file
# more like search engine
#search for particular text

def count_word_in_file(filename, word_to_count):
  try:
    with open(filename, "r") as file:
      content = file.read()
      count = content.lower().count(word_to_count.lower())
      return count
  except FileNotFoundError:
    print(f"File {filename} not found")

filename = "dummyfile.txt"
target_word = "python"

with open(filename, "w") as f:
  f.write("Python is a popular programming language. Python is versatile.\n")
  f.write("Learning Python can be very beneficial. Python is fun!")
  
occurences = count_word_in_file(filename, target_word)

if occurences != -1:
  print(f"The word {target_word} appears {occurences} times in {filename}")