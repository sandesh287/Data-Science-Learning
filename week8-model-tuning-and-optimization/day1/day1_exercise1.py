# Introduction to Hyperparameter Tuning
# Train a model with default hyperparameters, evaluate its performance, and manually adjust a few hyperparameters to observe their impact on results
# Dataset: Breast Cancer dataset



# libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display dataset info
print(f'Feature Names: {data.feature_names}')
print(f'Class Names: {data.target_names}')


# Train Random Forest model with default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)


# Predict and evaluate
y_pred_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

print(f'Default Model Accuracy: {accuracy_default}')
print('\nClassification Report (Default):\n', classification_report(y_test, y_pred_default))



# Train Random Forest model with adjusted hyperparameters
rf_tuned = RandomForestClassifier(
  n_estimators=400,
  max_depth=5,
  random_state=42
)
rf_tuned.fit(X_train, y_train)


# Predict and Evaluate
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f'Tuned Model Accuracy: {accuracy_tuned}')
print('\nClassification Report (Tuned):\n', classification_report(y_test, y_pred_tuned))


# Report
print('As this dataset is fully optimized, we cannot see any difference in our tuned model and default model. But while working on real-world dataset, we will obviously see the difference that tuned model is better than the default one.')