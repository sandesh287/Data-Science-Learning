# program that stores student grades in a dictionary and calculates the average grade

def calculate_averages(grades_dict):
  student_averages = {}  # empty dictionary for "name": average
  all_grades = []  # empty list to store every grade from all students
  
  for student, grades in grades_dict.items(): # .items return each key-value pair in dictionary
    if grades:  # ensure there are grades to calculate an average
      student_avg = sum(grades) / len(grades)
      student_averages[student] = student_avg
      all_grades.extend(grades)  # add all grades to a list for overall average
    else:
      student_averages[student] = 0.0  # assign 0 if no grades
    
  overall_average = sum(all_grades) / len(all_grades) if all_grades else 0.0
  return student_averages, overall_average

student_grades = {
  "Alice": [85, 90, 78, 92],
  "Bob": [70, 65, 80, 75],
  "Charlie": [95, 88, 91, 93],
  "David": []  #student with no grades
}

individual_averages, class_average = calculate_averages(student_grades)
print("Individual Student Averages: ")
for student, avg in individual_averages.items():
  print(f"{student}: {avg:.2f}")
  
print(f"\nOverall Class Average: {class_average:.2f}")