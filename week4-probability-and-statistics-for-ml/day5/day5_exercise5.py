# Use real-world datasets (eg. student scores by gender and class) for hypothesis testing
# Visualize test results using boxplots or bar plots


# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm


# Simulate Dataset
np.random.seed(42)
n_students = 50


# Factors
gender = np.random.choice(['male', 'female'], n_students*2)
students_class = np.random.choice(['A', 'B'], n_students*2)


# Simulate exam scores with slight differences
scores = (
  np.random.normal(75, 10, n_students*2) +   # base
  np.array([5 if g == 'female' else 0 for g in gender]) +   # female boost
  np.array([3 if c == 'B' else 0 for c in students_class])  # class B boost
)

df = pd.DataFrame({
  'score': scores,
  'gender': gender,
  'student_class': students_class
})


# Two-way ANOVA
model = ols('score ~ C(gender) + C(student_class) + C(gender):C(student_class)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("Two-way ANOVA Table:\n", anova_table)


# Interpretation
for factor in anova_table.index:
  p_value = anova_table.loc[factor, 'PR(>F)']
  if p_value < 0.05:
    print(f"{factor} is significant (p = {p_value:.4f})")
  else:
    print(f"{factor} is not significant (p = {p_value:.4f})")


# Visualization
# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='gender', y='score', hue='student_class', data=df)
plt.title("Boxplot: Student Scores by Gender and Class")
plt.ylabel("Exam Score")
plt.show()

# Bar plot
plt.figure(figsize=(8,5))
sns.barplot(x='gender', y='score', hue='student_class', data=df, errorbar=('ci', 95), capsize=0.1)
plt.title("Bar Plot (with 95% CI): Mean Exam Score by Gender and Class")
plt.ylabel("Mean Exam Score")
plt.show()