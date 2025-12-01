# Use Seaborn to create a violin plot or box plot for visualizing distributions

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# preparing datasets
data = {'Group': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
        'Score': [75, 82, 78, 85, 70, 88, 72, 80],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male']
      }
df = pd.DataFrame(data)

# creating box plot
plt.figure(figsize=(8, 6))  # adjust figure size
sns.boxplot(x='Group', y='Score', hue='Gender', data=df, palette='viridis', width=0.6)
plt.title("Scores by Group and Gender")
plt.xlabel("Group")
plt.ylabel("Score")
plt.show()


# Steps:
# 1. import libraries
# 2. prepare data: dataset containing numerical variables you want to visualize
# 3. create box plot using sns.boxplot()
# i. Single numerical variable: sns.boxplot(y=df['Value']) # vertical box plot of value
# ii. Numerical variable across categories: To compare distributions across different categories, specify both categorical variable(on x-axis) and numerical variable(on y-axis) syntax: sns.boxplot(x='Category', y='Value', data=df) # box plot of value for each category
# 4. Display the plot: plt.show()

# Customization options for sns.boxplot()
# orient: control the orientation('v': vertical, 'h': horizontal)
# color: set color of the boxes
# width: adjust width of boxes
# hue: create separate box plots for different levels of third categorical variable within each x-axis category
# palette: choose color palette for the plot
# whis: control length of whiskers
# outlier_kws: customize appearance of outliers

