# Naive Bayes
# Naive Bayes is a probabilistic classifier based on Bayes' theorem, which assumes that the features are conditionally independent given the class label. Despite this 'naive' assumption, it often performs well in text classification and spam detection tasks.




# libraries
from sklearn.naive_bayes import GaussianNB   # Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming that each feature follows a Gaussian (normal) distribution. It is particularly used for continuous data.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np



# Sample data (eg. hours studied and grades vs. pass/fail)
X = np.array([[1,50], [2,60], [3,55], [4,65], [5,70], [6,75], [7,80], [8,85], [9,90], [10,95]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # 0 = fail, 1 = pass. This is binary classification task, ideal for Gaussian Naive Bayes model.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train model
model = GaussianNB()   # initializes Gaussian Naive Bayes classifier model. This classifier will assume each feature is normally distributed and will calculate probabilities based on this assumption.

model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



# Display results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n {conf_matrix}')





# This code demonstrates a full workflow for using Gaussian Naive Bayes to classify students as passing or failing based on hours studied and grades. The model is evaluated using both accuracy and confusion matrix, providing a comprehensive view of its classification performance.