# Train and evaluate a Gradient Boosting model on a dataset, tune key parameters, and compare its performance with Random Forest Model
# Dataset: Breast Cancer Dataset, which is a binary classification dataset to predict whether a tumor is malignant or benign



# libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display dataset information
print(f'Features: {data.feature_names}')
print(f'Classes: {data.target_names}')


# Train Gradient Boosting Model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)


# Predict
y_pred_gb = gb_model.predict(X_test)


# Evaluate performance
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f'Gradient Boosting Accuracy: {accuracy_gb}')
print('\nClassification Report:\n', classification_report(y_test, y_pred_gb))


# Define Hyperparameter Grid
param_grid = {
  'learning_rate': [0.01, 0.1, 0.2],
  'n_estimators': [50, 100, 200],
  'max_depth': [3, 5, 7]
}


# Perform Grid Search
grid_search = GridSearchCV(
  estimator=GradientBoostingClassifier(random_state=42),
  param_grid=param_grid,
  cv=5,
  scoring='accuracy',
  n_jobs=-1
)

grid_search.fit(X_train, y_train)


# Display best parameters and score
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')


# Train the Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# Predict
y_pred_rf = rf_model.predict(X_test)


# Evaluate performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')


print(f'Since the accuracy is: {accuracy_rf} > {grid_search.best_score_} > {accuracy_gb} . Hence, Random Forest is still better as compared to others')