# concatenation
first = "hello"
second = "world"
result = first + " " + second
print(result)


# slicing
text = "Python Programming"
print(text[0:6])
print(text[-11:])


# formatting (f-formatting)
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old")


# common string method
# split()
sentence = "Python is fun"
words = sentence.split()
print(words) # displays ["Python", "is", "fun"]

sentences = "Python,is,fun"
word = sentences.split(",")
print(word)

# join()
new_sentence = " ".join(words) # there is space in " ", so joins with space
print(new_sentence) # prints:  Python is fun

# replace()
text = "I love Java"
updated_text = text.replace("Java", "Python")  # replace Java by Python
print(updated_text)

# strip()
messy = "       Hello, World      "
cleaned_text = messy.strip() # removes all the unnecessary spaces
print(cleaned_text)
