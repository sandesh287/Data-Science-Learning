# Compare k-NN results to Logistic regresion, analyzing accuracy and other metrics
# Objective: compare the k-NN and logistic regression model using the same dataset to analyze differences in performance



# libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


# Load Iris dataset
data = load_iris()
X, y = data.data, data.target


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)


# Predict using Logistic regression
y_pred_lr = log_reg.predict(X_test)


# Evaluate logistic regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy: ", accuracy_lr)


# Evaluate k-NN
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'k-NN Accuracy (k={best_k}): {accuracy_knn}')


# Detailed Comparison
print("\nLogistic Regression Classification Report: \n")
print(classification_report(y_test, y_pred_lr))

print("\nk-NN Report: \n")
print(classification_report(y_test, y_pred_knn))