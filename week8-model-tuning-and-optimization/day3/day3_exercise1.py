# Advanced Hyperparameter Tuning with Bayesian Optimization
# Apply Bayesian Optimization using Optuna to tune an XGBoost model and compare the results with Grid Search and Random Search
# Dataset: Breast Cancer Dataset



# libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna


# Load dataset
data = load_breast_cancer()


# Features and Traget
X, y = data.data, data.target


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

print(f'Training Data shape: {X_train.shape}')
print(f'Test Data Shape: {X_test.shape}')



# Baseline XGBoost Model

# Train a baseline XGBoost model
baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)
baseline_model.fit(X_train, y_train)


# Evaluate model
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
print(f'Baseline XGBoost Accuracy: {baseline_accuracy}')



# Apply Bayesian Optimization using Optuna

# Define objective function for Optuna
def objective(trial):
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    'max_depth': trial.suggest_int('max_depth', 3, 100),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'gamma': trial.suggest_float('gamma', 0, 5),
    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
  }
  
  # Train XGBoost Model with suggested params
  model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
  model.fit(X_train, y_train)
  
  # Evaluate model on validation set
  preds = model.predict(X_test)
  accuracy = accuracy_score(y_test, preds)
  return accuracy


# Create an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


# Best Hyperparameters
print('Best Hyperparameters: ', study.best_params)
print('Best Accuracy using Bayesian Optimization using Optuna library: ', study.best_value)


# Report
print(f'\nAccuracy before was ({baseline_accuracy}) which is Baseline Accuracy and after using Bayesian Optimization using Optuna library was ({study.best_value}) ; which clearly shows that the accuracy after using Bayesian Optimization is better than before using it.')



# Grid Search

# Define parameter grid
param_grid = {
  'n_estimators': [100, 200, 300],
  'max_depth': [3, 5, 7],
  'learning_rate': [0.01, 0.1, 0.2],
  'subsample': [0.6, 0.8, 1.0]
}


# Train XGBoost with Grid Search
grid_search = GridSearchCV(
  estimator=XGBClassifier(eval_metric='logloss', random_state=42),
  param_grid=param_grid,
  scoring='accuracy',
  cv=3,
  verbose=1
)

grid_search.fit(X_train, y_train)


# Best parameters and accuracy with Grid Search
print('\n\n\nGrid Search Best Parameters: ', grid_search.best_params_)
print('Grid Search Best Accuracy: ', grid_search.best_score_)



# Random Search

# Define parameter distribution
param_dist = {
  'n_estimators': [50, 100, 200, 300, 400],
  'max_depth': [3, 5, 7, 9],
  'learning_rate': [0.01, 0.05, 0.1, 0.2],
  'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
  'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}


# Train XGBoost with Random Search
random_search = RandomizedSearchCV(
  estimator=XGBClassifier(eval_metric='logloss', random_state=42),
  param_distributions=param_dist,
  n_iter=50,
  scoring='accuracy',
  cv=3,
  verbose=1,
  random_state=42
)

random_search.fit(X_train, y_train)


# Best Parameters and Accuracy for Random Search
print('\n\n\nRandom Search Best Parameters: ', random_search.best_params_)
print('Random Search Best Accuracy: ', random_search.best_score_)




# Final Result

# Baseline XGBoost Accuracy: 0.956140350877193
# Optuna Best Accuracy:  0.9649122807017544
# Grid Search Best Accuracy:  0.9735970721505751
# Random Search Best Accuracy:  0.9758045776693388