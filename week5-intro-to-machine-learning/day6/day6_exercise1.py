# k-NN (k-Nearest Neighbor) Algorithm

# Implement k-NN for a classification task, experimenting with different values of k
# Objective: To classify a dataset (eg. Iris Dataset) using k-NN algorithm



# libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load Iris dataset
data = load_iris()
X, y = data.data, data.target


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# Experiment with different values of k
for k in range(1, 11):
  
  # Initialize k-NN model
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  
  # Predict on test data
  y_pred = knn.predict(X_test)
  
  # Evaluate performance
  accuracy = accuracy_score(y_test, y_pred)
  print(f'k = {k}, Accuracy = {accuracy:.2f}')
