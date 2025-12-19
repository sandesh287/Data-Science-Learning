# Cross-Validation and Model Evaluation Techniques
# 1. Evaluate a classification model using K-Fold and Stratified K-Fold Cross-Validation
# 2. Compare the results to demonstrate the importance of stratification for imbalanced datasets
# Dataset: Credit card fraud detection



# libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)


# Display dataset info
print('Dataset Info:\n')
print(df.info)
print('\nClass Distribution:\n')
print(df['Class'].value_counts())


# Define features and target
X = df.drop(columns=['Class'])
y = df['Class']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Train and evaluate model using Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

scores_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

print(f'\n\nK-Fold Cross-Validation Scores: {scores_kfold}')
print(f'Mean Accuracy (K-Fold): {scores_kfold.mean()}')



# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Train and Evaluate model
scores_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

print(f'\n\nStratified K-Fold Cross-Validation Scores: {scores_stratified}')
print(f'Mean Accuracy (Stratified K-Fold): {scores_stratified.mean()}')



# Report
print('As we can see that accuracy for K-Fold is (0.999495270907854) and that for Statified K-Fold is (0.9995172156509907). So, its a little better when you use the stratified k fold as compared to k fold, especially for imbalanced dataset. And it maintains the reason, why stratified k fold cross validation maintains class distribution across folds, leading to more reliable and consistent performance metrics.')