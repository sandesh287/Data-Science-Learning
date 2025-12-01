# count the number of vowels in a string

def vowels_count(text):
  vowels = "aeiouAEIOU"
  vowel_count = 0
  
  for char in text:
    if char in vowels:
      vowel_count += 1
    
  return vowel_count

input_text = input("Enter the text: ")
print(vowels_count(input_text))