# EDA (Exploratory Data Analysis) Project

# Dataset: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Hypothesis: Higher class passenger have higher rate of survival

# Task 1: Perform Data Cleaning, Aggregation, and Filtering
# First load dataset and inspect its structure. Then handle the missing values, duplicates and inconsistencies. Then we will apply filters or aggregation to summarize specific groups

# Task 2: Generate Visualizations to Illustrate Key Insights
# create visualizations for trend, comparisons, and distributions in task 2

# Task 3: Identify and Interpret Patterns or Anamolies
# can use visualization and summary statistics to identify these patterns, and look for anamolies such as unusually high fares, or missing data trends

# Task 4: Summarize Findings in a Report
# create a report summarizing insights including key visualization and statistical summary

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1
# Load titanic dataset
dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(dataset_url)

# Inspect data
print(df.info())
print(df.describe())

# handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# remove duplicates
# just running this function drops duplicates, need not to do anything special
# having duplicates will cause problems in dataset
df = df.drop_duplicates()

# Filter data: Passengers in first class
first_class = df[df["Pclass"] == 1]
print("First Class Passengers:\n", first_class.head())


# Task 2
# Bar Chart: Survival rate by Class
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar", color="skyblue")
plt.title("Survival rate by Class")
plt.ylabel("Survival Rate")
plt.show()

# Histogram: Age Distribution
sns.histplot(df["Age"], kde=True, bins=20, color="purple")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Age vs Fare
plt.scatter(df["Age"], df["Fare"], alpha=0.5, color="green")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()


# Task 3
# Using summary statistics to identify initial patterns/anamolies
# Look for unusual maximum values or large standard deviations
# In output, a max value for exceeding the 75% percentile could indicate anamolies
print("Summary Statistics:\n", df['Fare'].describe())
# Use visualization to spot patterns and anamolies
plt.figure(figsize=(10, 6))

# a. Histogram to check distribution
sns.histplot(df['Fare'], kde=True, bins=20, color="red")
plt.title("Fare Distribution")
plt.xlabel("Fare in dollars ($)")
plt.ylabel("Frequency")
plt.show()
# An uneven distribution or spikes far from the main cluster suggest anomalies.

# b. Box plot to visualize outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title("Fare Box Plot (Outliers indicated by Diamonds)")
plt.xlabel("Fare in dollars ($)")
plt.show()
# Points plotted as individual markers beyond the "whiskers" are identified as outliers by the IQR method.

# Implement a Statistical Method (z-score) for Anamoly Detection
# The z-score method flags data points that are a certain number of standard deviations from the mean
# The z-score method quantifies how unusual a data point is. Data points beyond a certain standard deviation threshold (2.5 or 3) are flagged. This helps in identifying "unusually high fares" as requested
def z_score_anamoly_detection(data, threshold=3):
  mean = np.mean(data)
  std = np.std(data)
  z_scores = [(y - mean) / std for y in data]
  # flag anamolies based on the threshold
  anamolies = np.where(np.abs(z_scores) > threshold)[0]
  return anamolies, z_scores

anamoly_indices, z_scores = z_score_anamoly_detection(df['Fare'], threshold=2.5)

print("\n --- Z-Score Anamoly Detection Results ---\n")
print(f"Number of anamolies detected: {len(anamoly_indices)}")
print(f"Anamalous Fares:\n {df.loc[anamoly_indices]}")

# Visualizing the detected anamolies
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["Fare"], label="Normal Fares", color="yellow")
plt.scatter(anamoly_indices, df.loc[anamoly_indices]["Fare"], color='red', label='Anamoly')
plt.title("Detected Anamolies (Z-Score Method)")
plt.xlabel("Data Point Index")
plt.ylabel("Fare in dollars ($)")
plt.legend()
plt.show()


# Task 4
# 1. Overview
# Data Overview
print("Dataset Shape:\n", df.shape)
print("Missing values per column:\n", df.isna().sum())
print("Column names:\n", df.columns)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # fill missing age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # fill embarked with mode

print("\nMissing values after cleaning:\n", df.isnull().sum())

# 2. Key Insights
# Survival rate by class
survival_pclass = df.groupby('Pclass')['Survived'].mean() * 100
print('\nSurvival Rate by Class (%):\n', survival_pclass)

# Age distribution
print('\nAge Distribution:\n', df['Age'].describe())

# Correlation between Fare and Survival
fare_survival_corr = df['Fare'].corr(df['Survived'])
print('\nCorrelation between Fare and Survival:\n', fare_survival_corr)

# 3. Visual Insights
# Survival rate by class
plt.figure(figsize=(7,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger class')
plt.ylabel('Survival Rate')
plt.show()

# Age distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel("Age")
plt.show()

# Fare vs Survival
plt.figure(figsize=(7,4))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare vs Survival')
plt.show()

# Survival by Sex
plt.figure(figsize=(7,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()

# Survival by Embarkation
plt.figure(figsize=(7,4))
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Emabarkation Port')
plt.show()

# Pair plot
sns.pairplot(df)
plt.show()

# 4. Summary Text (Print out)
print("\n===== SUMMARY REPORT =====")
print("1. Overview:")
print("- Dataset contains 891 rows and 12 columns.")
print("- Missing values handled for 'Age' (median) and 'Embarked' (mode).")

print("\n2. Key Insights:")
print(f"- Highest survival rate: 1st class ({survival_pclass[1]:.0f}%)")
print(f"- Lowest survival rate: 3rd class ({survival_pclass[3]:.0f}%)")
print("- Most passengers are between ages 20–40.")
print(f"- Fare and survival correlation: {fare_survival_corr:.2f} (positive correlation)")

print("\n3. Visual Insights:")
print("- PNG images saved for:")
print("  * Survival rate by class")
print("  * Age distribution")
print("  * Fare vs survival")
print("  * Survival by sex")
print("  * Survival by embarkation")

# # Basic Statistics
# print("\nSurvival Counts:\n")
# print(df["Survived"].value_counts())
# print("\nSurvival Rate by Sex:\n")
# print(df.groupby('Sex')['Survived'].mean())
# print("\nSurvival Rate by Pclass:\n")
# print(df.groupby('Pclass')['Survived'].mean())
# print("\nAge Statistics:\n")
# print(df['Age'].describe())
# print("\nFare Statistics:\n")
# print(df['Fare'].describe())

# # Feature Engineering
# df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# # Categorize family size
# df['FamGroup'] = pd.cut(df['FamilySize'], bins=[0,1,4,20], labels=['Alone', 'Small', 'Large'])

# # Visualizations
# plt.figure(figsize=(8,5))
# sns.barplot(x='Sex', y='Survived', data=df)
# plt.title("Survival Rate by Sex")
# # plt.savefig('survival_by_sex.png')  # save image to current folder
# # plt.close()  # close image
# plt.show()

# plt.figure(figsize=(8,5))
# sns.barplot(x='Pclass', y='Survived', data=df)
# plt.title('Survival Rate by Passenger Class')
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(x='Survived', y='Age', data=df)
# plt.title('Age Distribution by Survival')
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(x='Survived', y='Fare', data=df)
# plt.title('Fare Distribution by Survival')
# plt.show()

# plt.figure(figsize=(8,5))
# sns.barplot(x='Embarked', y='Survived', data=df)
# plt.title('Survival Rate by Embarkation Port')
# plt.show()

# plt.figure(figsize=(8,5))
# sns.barplot(x='FamGroup', y='Survived', data=df)
# plt.title('Survival Rate by Family Size Group')
# plt.show()

# # Save summary statistics as a Report
# summary_stats = df.describe(include='all')
# summary_stats.to_csv('titanic_summary_stats.csv')
# print("Summary statistics saved to titanic_summary_stats.csv file!")