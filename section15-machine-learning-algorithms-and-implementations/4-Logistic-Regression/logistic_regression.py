# Logistic Regression
# Logistic Regression is a supervised learning algorithm used for binary classification problems (eg. yes/no, spam/not spam). Instead of predicting a continuous output, it predicts the probability belonging to a particular class by applying the logistic (sigmoid) function, which outputs values between 0 and 1.




# libraries
from sklearn.linear_model import LogisticRegression   # to create a logistic regression model for predicting binary target value
from sklearn.model_selection import train_test_split   # to split the dataset into training and testing sets, helping evaluate model's performance on unseen data
from sklearn.metrics import accuracy_score, confusion_matrix   # accuracy_score calculates accuracy between actual and predicted values, to evaluate classification models, where higher values indicates better performance, confusion_matrix provides detailed breakdown of model's classification performance by showing true positives, true negatives, false positives, and false negatives
import numpy as np   # fundamental library for numerical computations



# Sample data (eg. hours studied vs. pass/fail)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])   # based on hours studied, defines the feature data X as numpy array, where each element is a list containing a single number representing hours studied by student. Shape: (10, 1), meaning there are 10 example rows with 1 feature column
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # defines the target data y as numpy array containing pass/fail status. Here, 0 indicates fail and 1 indicates pass. This is binary classification task, making it well suited for logistic regression.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train the model
model = LogisticRegression()  # finds the best fitting logistic or S-shaped curve that separates the binary outcomes

model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)   # calculates the accuracy of the model by comparing the actual values (y_test) to predicted values (y_pred). Accuracy is the ratio of correctly predicted instances to the total instances.

conf_matrix = confusion_matrix(y_test, y_pred)  # computes confusion matrix for the model, showing a breakdown of true positives, true negatives, false positives, and false negatives. This helps to understand the model's classification performance in more detail.



# Display
print(f'\nAccuracy: {accuracy}')
print(f'\nConfusion Matrix:\n {conf_matrix}')





# This code provides a complete example of a logistic regression workflow, including data preparation, model training, prediction, and performance evaluation. This is particularly useful for binary classification task, where the goal is to predict one of the two outcomes, in this case fail vs pass.