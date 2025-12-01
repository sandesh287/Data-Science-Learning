# example 1: checking a condition
num = 10
if num > 0:
  print("Positive number")
elif num == 0:
  print("Zero")
else:
  print("Negative number")
  
# example 2: Nested conditions
age = 25
if age > 18:
  if age < 30:
    print("Young adult")
  else:
    print("Adult")
    
# loop through a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
  print(fruit)
  
# loop with range
for i in range(5):  # create list of [0,1,2,3,4]
  print(i)
  
# count down from 5 using while-loop
count = 5
while count > 0:
  print(count)
  count -= 1
  
print("Outside while loop")

# example of break
for i in range(10):
  if i == 5:
    break
  print(i)

print("Outside for-loop")

# example of continue
for i in range(10):
  if i == 5:
    continue
  print(i)

print("Outside for-loop")

for i in range(10):
  if i % 2 == 0:
    continue
  print(i)