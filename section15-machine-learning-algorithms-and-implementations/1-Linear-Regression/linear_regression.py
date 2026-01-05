# Linear Regression
# Linear Regression is a supervised learning algorithm used for predicting a continuous target variable based on one or more input features. It finds the line of best fit (linear relationship) by minimizing the sum of squared differences between the actual and predicted values.




# libraries
from sklearn.linear_model import LinearRegression    # to create a linear regression model for predicting continuous target value
from sklearn.model_selection import train_test_split   # to split the dataset into training and testing sets, helping evaluate model's performance on unseen data
from sklearn.metrics import mean_squared_error   # calculates MSE between actual and predicted values. to evaluate regression models, where lower values indicates better performance
import numpy as np   # fundamental library for numerical computations



# Sample data (eg. house size vs. house price)
X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450], [1425], [1700]])   # based on house sizes, defines the feature data X as numpy array, where each element is a list containing a single number representing house size in square feet. Shape: (10, 1), meaning there are 10 example rows with 1 feature column
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])   # defines the target data y as numpy array containing house prices. Each value in y represents the price corresponding to house size in X.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # here, we get (X_train, y_train), which are training data model uses to learn, (X_test, y_test), are testing data uses to evaluate the model's performance; test_size=0.2 : 20% of data should be used for testing and 80% for training; random_state=42 : This seed ensures that each time you run the code, the same split of training and testing data is generated.



# Initialize and train the model
model = LinearRegression()   # creates an instance of linear regression class which represents a simple linear regression

model.fit(X_train, y_train)   # trains the model on training data. During this process, model learns the relationship between house size and house price by finding the best fit line that minimizes the error in predecting y_train from X_train



# Make predictions
y_pred = model.predict(X_test)   # The predict method uses the trained model to output estimated house prices, which are y_pred for the house sizes in X_test



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)   # calculates the mean squared error between actual values y_test and predicted values y_pred. MSE measures how well the model's predictions match the true values, with lower values indicating better performance.



# Display
print(f'\nMean Squared Error: {mse}')
print(f'\nPredicted Values: {y_pred}')




# This code demonstrate full workflow for training anf evaluating a simple linear regression model in Python from data preparation, data training, prediction and performance evaluation.