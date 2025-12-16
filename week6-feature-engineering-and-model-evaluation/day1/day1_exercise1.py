# Introduction to Feature Engineering

# 1. Load a dataset and explore its feature, identifying categorical and numerical features
# 2. Plan which feature engineering techniques might be most suitable for the dataset
# Objective: To explore and analyze a dataset to identify different types of features and plan appropriate feature engineering technique


# libraries
import pandas as pd


# load titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)


# Display dataset information
print("Dataset Information: \n", df.info())


# Preview the first few rows
print("\nDataset Preview:\n", df.head())


# Separate features by their data types
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

print('\nCategorical Features: ', categorical_features.tolist())
print('\nNumerical Features: ', numerical_features.tolist())


# Task 2: Plan Feature Engineering

# Display summary of categorical features
print('\nCategorical Feature Summary: \n')
for col in categorical_features:
  print(f'{col}:\n', df[col].value_counts(), '\n')
  

# Display summary of numerical features
print('\nNumerical Features Summary: \n')
print(df[numerical_features].describe())



""" 
For gender we have two different values: Male and Female, So will apply One-hot Encoding for Sex, because it can have binary values (0, 1) for Male or Female

For Embarked, we will fill missing values and encode the categories, because Embarked has S, C & Q

For ordinal features which is Pclass, because encode as ordinal values one is less than two is less than three, because the class is the class of the passenger first class, second class or third class. So we have 891 of them, but it displays the values based on the class here.

There can be two different deliverables:
  1. Feature analyst report where we list categorical, numerical and ordinal features and summary statistics for each feature type and for feature.
  2. Feature engineering plan with clear documentation of pre-processing steps for each feature.
"""