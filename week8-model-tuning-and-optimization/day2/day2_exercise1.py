# Grid Search and Random Search
# Implement both Grid Search and Random Search for hyperparameter tuning, compare their efficiency, and analyze the impact on model performance
# Dataset: Iris Dataset, multiclass classification problem



# libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# Load dataset
data = load_iris()


# Features and target
X, y = data.data, data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Display dataset info
print(f'Feature Names: {data.feature_names}')
print(f'Class Names: {data.target_names}')


# We will define grid of hyperparameters for Random Forest Model and perform exhaustive search using GridSearchCV

# Define hyperparameter grid with random values
param_grid = {
  'n_estimators': [50, 100, 150],
  'max_depth': [None, 5, 10],
  'min_samples_split': [2, 5, 10]
}


# Initialize Grid Search
grid_search = GridSearchCV(
  estimator=RandomForestClassifier(random_state=42),
  param_grid=param_grid,
  cv=5,   # cross-validation
  scoring='accuracy',
  n_jobs=-1  # -1 means use all the processor
)


# Perform grid search
grid_search.fit(X_train, y_train)


# Predict and Evaluate Best model
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f'Best Hyperparameters (Grid Search): {grid_search.best_params_}')
print(f'Grid Search Accuracy: {accuracy_grid}')



# Random Search

# Define hyperparameter distribution
param_dist = {
  'n_estimators': np.arange(50, 200, 10),
  'max_depth': [None, 5, 10, 15],
  'min_samples_split': [2, 5, 10, 20]
}


# Initialize Random Search
random_search = RandomizedSearchCV(
  estimator=RandomForestClassifier(random_state=42),
  param_distributions=param_dist,
  n_iter=20,
  cv=5,
  scoring='accuracy',
  n_jobs=-1,
  random_state=42
)


# Perform Random Search
random_search.fit(X_train, y_train)


# Evaluate best model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f'Best Hyperparameters (Random Search): {random_search.best_params_}')
print(f'Random Search Accuracy: {accuracy_random}')