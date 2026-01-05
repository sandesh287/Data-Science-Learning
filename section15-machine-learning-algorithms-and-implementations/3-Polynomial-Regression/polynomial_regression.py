# Polynomial Regression
# Polynomial Regression is an extension of Linear Regression that models the relationship between the input features and the target variable as an nth-degree polynomial. It can capture non-linear relationships in the data by adding polynomial terms to the features.




# libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures   # used to generate polynomial features of a specified degree. This transformation allows a linear regression model to fit a non-linear relationship by adding polynomial terms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np



# Sample Data (eg. experience vs. salary)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 400000, 500000])
# The relationship between experience and salary is non-linear, making it suitable for polynomial regression



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Transform features into polynomial features
poly = PolynomialFeatures(degree=2)   # initializes a polynomial feature object with degree 2. This means we will transform the feature X into a second degree polynomial, adding squared terms to represent the non-linear relationship.

X_train_poly = poly.fit_transform(X_train)   # transforms the X_train data into a polynomial features for degree 2. This transformation will add an aditional column representing x² alongside the original feature, allowing a linear regression  model to fit a quadratic curve to the data.

X_test_poly = poly.transform(X_test)   # transforms the X_test data into polynomial features using the same transformation learned from X_train. This ensures that the test data is represented in the same polynomial feature space as the training data. We only call transform here, not fit_transform, to ensure consistency with the training data transformation.



# Initialize and train the model
model = LinearRegression()   # fits a linear relationship to polynomial transformed data
model.fit(X_train_poly, y_train)   # trains the model on polynomial transformed training data (X_train_poly) and target values (y_train). The model learns the relationship between experience in polynomial form and salary, allowing it to capture non-linear patterns.



# Make predictions
y_pred = model.predict(X_test_poly)   # make predictions on test set (X_test_poly). These predictions are based on polynomial features of (X_test), allowing the model to output salary estimates based on years of experience



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'Predicted Values: {y_pred}')





# This code demonstrates how to use a polynomial regression to fit a nonlinear relationship between years of experience and salary. By transforming the features into polynomial terms, we allow a simple linear regression model to fit a curve to the data.