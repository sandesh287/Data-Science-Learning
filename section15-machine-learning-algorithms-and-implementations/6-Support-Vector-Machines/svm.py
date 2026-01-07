# Support Vector Machines (SVM)
# Support Vector Machines (SVM) is a powerful classification algorithm that works by finding the hyperplace that best separates classes in the feature space. SVM aims to maximize the margin between the classes, making it a good choice for binary classification, especially when classes are well-separated.




# libraries
from sklearn.svm import SVC   # support vector classifier, used for classification task and aims to find the optimal hyperplace that separates classes in feature space
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np



# Sample Data (eg. hours studied and grades vs. pass/fail)
X = np.array([[1,50], [2,60], [3,55], [4,65], [5,70], [6,75], [7,80], [8,85], [9,90], [10,95]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # 0 = fail, 1 = pass. This is binary classification task, suitable for SVM.



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train the SVM model
model = SVC(kernel='linear')   # initializes the SVM classifier with a linear kernel, kernel='linear' specifies that we want to use a linear SVM, which attempts to find a linear hyperplane that best separates the two classes. Other kernel options include 'rbf' (radial basis function) and 'poly' (polynomial), which can capture non-linear relationships.

model.fit(X_train, y_train)   # The model learns and the optimal hyperplane that separates the data points belonging to each class, maximizing the margin between them.



# Make predictions
y_pred = model.predict(X_test)   # The model classifies each data point in (X_test) based on which side of the hyperplane it falls on. The output (y_pred) contains the predicted classes, which is 0 or 1 for each data point in (X_test).



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)





# This code here demonstrates a complete implementation of the Support Vector Machine classifier, showing how it uses our studied and prior grades to classify students as passing or failing. The SVM model is trained, tested, and evaluated using both accuracy and confusion matrix.