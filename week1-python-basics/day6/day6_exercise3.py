# program to copy the contents of one file to another
# open, read, write to another file

def read_file(filename1, filename2):
  try:
    with open(filename1, "r") as firstfile, open(filename2, 'w') as secondfile:
      for line in firstfile:
        secondfile.write(line)
  except FileNotFoundError:
    print(f"File {filename1} not found")

read_file("fruits.txt", "secondfruits.txt")


# OR (more simple way)
import shutil
shutil.copyfile('fruits.txt', 'copy_fruits.txt')