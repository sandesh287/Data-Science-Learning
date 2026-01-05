# K-Nearest Neighbors (KNN)
# K-nearest Neighbors (KNN) is a simple, non-parametric classification (or regression) algorithm. It classifies new data points based on the majority class of the k-nearest points in the feature space. It is particularly useful for small datasets where the relationships among data points can be easily visualized.
# KNN is non-parametric model, meaning it does not learn a specific set of parameters. Instead, it stores the training data to make predictions based on the nearest neighbors.




# libraries
from sklearn.neighbors import KNeighborsClassifier   # for classification task, where the model classifies data points based on the nearest data points
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np



# Sample data (eg. hours studied and prior grades vs. pass/fail)
X = np.array([[1,50], [2,60], [3,55], [4,65], [5,70], [6,75], [7,80], [8,85], [9,90], [10,95]])   # features: [hours studied, prior grades]. X has shape (10, 2), indicating 10 samples with 2 features each
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # labels: 0 = fail, 1 = pass. This is a binary classification task, where the model predicts whether a student will pass or fail based on the hours studied and prior grades.



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train the KNN model with k=3
model = KNeighborsClassifier(n_neighbors=3)   # initializes the knn classifier model with k=3, meaning the model will classify a new data point based on a majority class among its three nearest neighbors. You can increase it or reduce it.

model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)   # The model classifies each data point in (X_test) based on the majority class of its 3 nearest neighbors from the training set. The output (y_pred) contains the predicted classes, which is 0 or 1 for each data point in (X_test).



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)





# This code demonstrates a full workflow for training and evaluating a K-Nearest Neighbor classifier. The KNN model predicts binary outcomes based on hours studied and prior grades, showing how it classifies each test data point by looking at the classes of the nearest neighbors.