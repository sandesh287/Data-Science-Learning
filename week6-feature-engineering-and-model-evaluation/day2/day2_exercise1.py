# 1. Apply Min-Max Scaling and standardization to a dataset using scikit-learn
# 2. Observe the effects of scaling on model performance by training a k-NN classifier before and after scaling
# Objective: To apply min-max scaling and standardization to the Iris Dataset using scikit-learn, compare the performance of k-NN classifier before and after scaling



# libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Load iris dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# Display dataset information
print('\nDataset Information: \n', X.describe())
print('\nTarget Classes: \n', data.target_names)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Predict and Evaluate
y_pred = knn.predict(X_test)
print("\nAccuracy without Scaling: ", accuracy_score(y_test, y_pred))



# Apply Min-Max Scaling
scalar = MinMaxScaler()
X_scaled = scalar.fit_transform(X)


# Split scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train k-NN classifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)


# Predict and evaluate
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print('\nAccuracy with Min-Max Scaling: ', accuracy_score(y_test_scaled, y_pred_scaled))



# Apply Standardization
scalar_std = StandardScaler()
X_stand = scalar_std.fit_transform(X)


# Split scaled data
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_stand, y, test_size=0.2, random_state=42)


# Train k-NN classifier on standardized data
knn_stand = KNeighborsClassifier(n_neighbors=5)
knn_stand.fit(X_train_std, y_train_std)


# Predict and evaluate
y_pred_std = knn_stand.predict(X_test_std)
print('\nAccuracy with Standardization: ', accuracy_score(y_test_std, y_pred_std))