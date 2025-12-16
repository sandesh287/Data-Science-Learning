# Supervised Learning Mini Project

# Mini Project: Building a Supervised Learning Model
# Objective: To use a dataset to develop, train and evaluate regression and classification models
# For demonstration, we will use 2 datasets: California Housing Dataset, predicting the house prices for regression; Telco Customer Churn Dataset, predicting customer churn for classification
# Tasks:
  # 1. Perform Exploratory Data Analysis (EDA) and Preprocessing
  # 2. Train and Evaluate Multiple Models
  # 3. Summarize Findings in a Report (Compare the metrics such as MSE for regression and accuracy, f1 score for classification and highlight the best performing models and discuss possible reasons for their success)



# Task 1: Perform EDA and Preprocessing
# Task 2: Train and Evaluate Multiple Models

# California Housing Dataset
# libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Regression
# Load California Housing Dataset
data = fetch_california_housing(as_frame=True)
df_housing = data.frame


# Inspect data
print(df_housing.info())
print(df_housing.describe())


# Visualize relationships
sns.pairplot(df_housing, vars=['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal'])
plt.show()


# Check for missing values
print('Missing Values: \n', df_housing.isnull().sum())


# Define features and target
X = df_housing[['MedInc', 'HouseAge', 'AveRooms']]
y = df_housing['MedHouseVal']


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression MSE: ", mse)



# Task 3: Summarize Findings in a Report

report = f"""
MODEL PERFORMANCE REPORT
========================

1. California Housing Dataset (Regression)
------------------------------------------
Model Used: Linear Regression
Evaluation Metric: Mean Squared Error (MSE)

MSE: {mse:.4f}

The Linear Regression model performs well due to the linear
relationship between income, house age, rooms, and house value."""


with open('model_report.txt', 'w') as f:
  f.write(report)