# Perform Chi-Square Test
# To test the independence of two categorical variables


# importing libraries
from scipy.stats import chi2_contingency

# Contingency Table
data = [[50, 30, 20], [30, 40, 30], [20, 30, 40]]

# Perform Chi-Square Test
chi2, p_value, dof, expected = chi2_contingency(data)

print("Chi-Square Statistic: ", chi2)
print("P-Value: ", p_value)
print("Expected Frequencies: \n", expected)