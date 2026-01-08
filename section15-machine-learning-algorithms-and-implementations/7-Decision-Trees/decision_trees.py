# Decision Trees
# Decision Trees are a versatile supervised learning algorithm used for both classification and regression. They work by recursively splitting the data into subsets based on the feature that provides the most information gain. Each node represents a decision based on a feature, and each leaf node represents a prediction.




# libraries
from sklearn.tree import DecisionTreeClassifier   # to create decision tree model, which makes predictions by learning simple decision rules from data features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np



# Sample data (eg. hours studied and grades vs. pass/fail)
X = np.array([[1,50], [2,60], [3,55], [4,65], [5,70], [6,75], [7,80], [8,85], [9,90], [10,95]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # 0 = fail, 1 = pass. This is binary classification task, ideal for decision tree classifier.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train the model
model = DecisionTreeClassifier()   # initializes a decision tree classifier model. By default, the classifier will split data based on the Gini impurity metric to decide the best splits at each node. Other splitting criteria such as entropy are also available.

model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



# Display results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: {conf_matrix}')





# The code provides the complete workflow for training and evaluating a decision tree classifier on data representing studied and prior grades. The decision tree model predicts whether a student will pass or fail based on input data and the evaluation metrics, which are accuracy and confusion matrix provide insights into its performance.