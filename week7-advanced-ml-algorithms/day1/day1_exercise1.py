# Build a basic ensemble model combining predictions from Linear Regression, Decision Tree, and k-NN to observe the impact on accuracy
# Dataset: Iris dataset to classify flowers into different species



# libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# Load Dataset
data= load_iris()


# Extracting features and target
X, y = data.data, data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train individual Models
log_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)


# Creating Voting Classifer
ensemble_model = VotingClassifier(
  estimators=[
    ('log_reg', log_model),
    ('decision_tree', dt_model),
    ('knn', knn_model)
  ],
  voting='hard'
)


# Train the ensemble Model
ensemble_model.fit(X_train, y_train)


# Predict with ensemble
y_pred_ensemble = ensemble_model.predict(X_test)


# Evaluate accuracy of ensemble
accuracy = accuracy_score(y_test, y_pred_ensemble)


# Evaluate individual models
y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)


# Displaying accuracy scores
print(f'Ensemble Model Accuracy: {accuracy:.2f}')
print(f'Logistic Regression Model Accuracy: {accuracy_score(y_test, y_pred_log):.2f}')
print(f'Decision Tree Model Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}')
print(f'k-NN Model Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}')