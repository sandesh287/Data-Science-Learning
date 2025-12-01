# reverse the words in a sentence(not the letters)

def reverse_sentence_words(sentence):
  words = sentence.split()
  reversed_words = words[::-1]
  reversed_sentence = " ".join(reversed_words)
  return reversed_sentence

input_text = input("Enter a sentence: ")
output_text = reverse_sentence_words(input_text)
print("Reversed sentence: ", output_text)