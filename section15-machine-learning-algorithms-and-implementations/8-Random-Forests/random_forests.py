# Random Forests
# Random Forests are an ensemble learning method that combines multiple decision trees to make a more accurate and stable predictions. Each tree in the forest is trained on a random subset of the data, and the final prediction is made by averaging (for regression) or voting (for classification) the predictions of individual trees. This helps to reduce overfitting and improve generalization.




# libraries
from sklearn.ensemble import RandomForestClassifier   # ensemble learning method that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np



# Sample data (eg. hours studied and grades vs. pass/fail)
X = np.array([[1,50], [2,60], [3,55], [4,65], [5,70], [6,75], [7,80], [8,85], [9,90], [10,95]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # 0 = fail, 1 = pass. This is binary classification task, suitable for random trees classifier.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)   # initializes random forest classifier with parameters: n_estimators=100 specifies the number of decision trees in the forest. The model builds 100 decision trees, each trained on random subsets of data. A higher number of trees generally increases accuracy, but it also requires more computation. And, random_state=42 ensures reproducibility, meaning each run will produce the same results.

model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



# Display results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n {conf_matrix}')






# This code here demonstrate a complete workflow for training and evaluating a random forest classifier, which predicts whether a student will pass or fail based on the hours studied and grades. The model is evaluated using both accuracy and confusion matrix, offering a detailed assessment of its performance.