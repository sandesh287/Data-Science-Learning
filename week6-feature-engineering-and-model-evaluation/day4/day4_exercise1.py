# 1. Use correlation and mutual information to select important features from a dataset
# 2. Apply tree-based model (Eg. Random Forest) to identify the most important features
# Onjective: To use the filter and embedded methods, to select important features from a dataset and compare the results
# Dataset: Diebeties Dataset from scikit-learn library



# libraries
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# Load dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


# Display dataset information
print(df.head())
print(df.info())


# Calculate the correlation matrix
correlation_matrix = df.corr()


# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Select Features with high correlation to the target
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
print('Features Most Correlated with Target: \n')
print(correlated_features)


# Separate the features and target
X = df.drop(columns=['target'])
y = df['target']


# Calculate Mutual Information
mutual_info = mutual_info_regression(X, y)


# Create a dataframe for better visualization
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

print('Mutual Information Scores:')
print(mi_df)



# Feature Selection using Tree-based model

# libraries
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# Train a Random Forest Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)


# Get feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print('Feature Importance from Random Forest: \n')
print(importance_df)


# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance From Random Forest')
plt.show()