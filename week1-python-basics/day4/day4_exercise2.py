# word frequency counter
# takes a sentence and returns a dictionary with frequency count of each word

sentence = input("Enter a sentence: ")

# split the sentence into words
words = sentence.split()

# initialize a dictionary
word_count = {}

# count the word frequency
for word in words:
  word = word.lower()
  if word in word_count:
    word_count[word] += 1
  else:
    word_count[word] = 1

print(word_count)

# prints: Enter a sentence: The AI Engineer is the engineer of the future who would work on AI projects and build amazing tools
# {'the': 3, 'ai': 2, 'engineer': 2, 'is': 1, 'of': 1, 'future': 1, 'who': 1, 'would': 1, 'work': 1, 'on': 1, 'projects': 1, 'and': 1, 'build': 1, 'amazing': 1, 'tools': 1}