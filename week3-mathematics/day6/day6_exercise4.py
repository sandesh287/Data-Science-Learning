# Perform hypothesis testing on real-world datasets (eg., comparing exam scores of two groups)

# importing libraries
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind
import matplotlib.pyplot as plt


# Simulated exam scores (sample data)
group_A = np.array([78, 85, 92, 88, 76, 81, 95, 89])
group_B = np.array([82, 79, 88, 84, 80, 77, 90, 85])


# Checking assumptions
# For t-test (independent samples), 
# 1. data should be appoximately normally distributed (Shapiro-walk tesst can check this). 
# 2. Variance should be similar (Levene's test can check this).
# If p-value > 0.05, assumption is satisfied
# Normality test
print("Group A normality p-value: ", shapiro(group_A).pvalue)
print("Group B normality p-value: ", shapiro(group_B).pvalue)

# Equal variance test
print("Variance equality p-value: ", levene(group_A, group_B).pvalue)


# Perform independent two-sample t-test
t_stat, p_value = ttest_ind(group_A, group_B)
print("T-Statistic: ", t_stat)
print("P-Value: ", p_value)

# Conclusion
alpha = 0.05
if p_value < alpha:
  print("Reject H0 → Significant difference between groups")
else:
  print("Failed to reject H0 → No significant difference between groups")


# Visualize the data
plt.boxplot([group_A, group_B], tick_labels=['Group A', 'Group B'])
plt.title('Exam Scores Comparison')
plt.ylabel('Scores')
plt.show()