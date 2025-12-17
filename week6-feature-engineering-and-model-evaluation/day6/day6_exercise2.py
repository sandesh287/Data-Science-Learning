# 2. Regression Model Evaluation
# Objective: Train a regression model and evaluate its performance using MAE, MSE, and R2
# Dataset: California Housing Dataset to predict house prices



# libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = fetch_california_housing()


# Extracting features and target
X, y = data.data, data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluate Regression matrix
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Display
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-Squared: {r2:.2f}')