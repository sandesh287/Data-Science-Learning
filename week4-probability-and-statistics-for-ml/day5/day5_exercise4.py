# Perform a two-way ANOVA to test for interaction effects
# We will use real-world dataset (tips from Seaborn)
# Factors: Sex, Smoker | Response variable: total_bill


# importing libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = sns.load_dataset('tips')

# Fit two-way ANOVA model with interaction
model = ols('total_bill ~ C(sex) + C(smoker) + C(sex):C(smoker)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA

print('\nTwo-way ANOVA Table:\n', anova_table)

# Interpretation
for factor in anova_table.index:
  p_value = anova_table.loc[factor, 'PR(>F)']
  if p_value < 0.05:
    print(f"{factor} is significant (p = {p_value:.4f})")
  else:
    print(f"{factor} is not significant (p = {p_value:.4f})")
    
# Visualization
# Point Plot
plt.figure(figsize=(8,5))
sns.pointplot(x='sex', y='total_bill', hue='smoker', data=df, dodge=True, markers=['o', 's'], capsize=0.1)
plt.title("Interaction Plot: Total Bill by Sex and Smoker")
plt.ylabel("Mean Total Bill")
plt.show()

# BoxPlot: Total bill by sex and smoker
plt.figure(figsize=(8,5))
sns.boxplot(x='sex', y='total_bill', hue='smoker', data=df)
plt.title("Boxplot: Total Bill by Sex and Smoker")
plt.ylabel("Total Bill")
plt.show()

# Bar plot: Mean Total bill by sex and smoker
plt.figure(figsize=(8,5))
sns.barplot(x='sex', y='total_bill', hue='smoker', data=df, ci=95, capsize=0.1)
plt.title("Bar Plot (with 95% CI): Mean Total Bill by Sex and Smoker")
plt.ylabel("Mean Total Bill")
plt.show()