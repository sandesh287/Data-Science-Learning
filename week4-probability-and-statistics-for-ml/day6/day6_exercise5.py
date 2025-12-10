# use real-world datasets (eg. housing prices) for regression analysis


# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Features and target
X = df[['MedInc', 'AveRooms', 'Population']]
y = df['MedHouseVal']


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.25, random_state=42
)


# 3. Fit multiple linear regression
model = LinearRegression()
model.fit(X_train, y_train)


# 4. Predictions
y_pred = model.predict(X_test)


# 5. Model Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Intercept:", model.intercept_)
print("Slope:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef}")

print("\nR-squared:", r2)
print("Root Mean Square Error:", rmse)


# Visualization (Income vs Price)
plt.figure(figsize=(8,5))
plt.scatter(df['MedInc'], y, alpha=0.4, label="Actual Data")
plt.plot(
    df['MedInc'],
    model.predict(df[['MedInc', 'AveRooms', 'Population']]),
    color='red',
    label="Regression Prediction"
)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Housing Prices vs Median Income")
plt.legend()
plt.show()