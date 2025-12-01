# create basic plots with Matplotlib
# objective: To plot a line graph, bar chart and a scatter plot
# create data for trends, categorical comparisons & relationships and then we are gonna customize plots with titles, labels and legends

# importing libraries
import matplotlib.pyplot as plt

# creating dataset
years = [2010, 2011, 2012, 2013]
sales = [100, 120, 140, 160]

# Line Plot
plt.plot(years, sales, label="Sales Trend", color="blue", marker="o")
plt.title("Sales over Years")
plt.xlabel("Years")
plt.ylabel("Sales")
plt.legend()
plt.show()

# creating dataset
categories = ["Electronics", "Clothing", "Groceries"]
revenue = [250, 400, 150]

# Bar Chart
plt.bar(categories, revenue, color="green")
plt.title("Revenue by Category")
plt.show()

# creating dataset
hours_studied = [1, 2, 3, 4, 5]
exam_scores = [50, 55, 65, 70, 85]

# Scatter plot
plt.scatter(hours_studied, exam_scores, color="red")
plt.title("Study hours vs Exam scores")
plt.xlabel("Hours studied")
plt.ylabel("Exam scores")
plt.show()