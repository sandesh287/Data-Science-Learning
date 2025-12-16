# Mini Project: Building a Supervised Learning Model
# Objective: To use a dataset to develop, train and evaluate regression and classification models
# For demonstration, we will use 2 datasets: California Housing Dataset, predicting the house prices for regression; Telco Customer Churn Dataset, predicting customer churn for classification
# Tasks:
  # 1. Perform Exploratory Data Analysis (EDA) and Preprocessing
  # 2. Train and Evaluate Multiple Models
  # 3. Summarize Findings in a Report (Compare the metrics such as MSE for regression and accuracy, f1 score for classification and highlight the best performing models and discuss possible reasons for their success)



# Task 1: Perform EDA and Preprocessing
# Task 2: Train and evaluate multiple models

# Telco Customer Churn Dataset
# libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


# Classification
# Load Telco Customer Churn Dataset
df_telco = pd.read_csv('telco_customer_churn.csv')


# Inspect the data
print(df_telco.info())
print(df_telco.describe())


# Visualize Churn distribution
sns.countplot(x='churn', data=df_telco)
plt.title('Churn Distribution')
plt.show()


# Separate numeric and categorical columns
categorical_cols = ['gender', 'contract_type', 'payment_method']
numeric_cols = ['customer_id', 'age', 'tenure', 'monthly_charges', 'total_charges']


# One-hot encoding categorical columns
df_telco = pd.get_dummies(df_telco, columns=categorical_cols, drop_first=True)


# Handle Missing vales
df_telco.fillna(df_telco.mean(), inplace=True)


# Encode categorical variable
le = LabelEncoder()
df_telco['churn'] = le.fit_transform(df_telco['churn'])


# Define features and target
X = df_telco.drop(columns=['churn'])
y = df_telco['churn']


# Scale Features
scalar = StandardScaler()
# X[numeric_cols] = scalar.fit_transform(X[numeric_cols])
X = scalar.fit_transform(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)


# Train k-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


# Evaluate models
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print('\nLogistic Regression Classification Report: \n')
print(classification_report(y_test, log_pred))

print('\nk-NN Classification Report: \n')
print(classification_report(y_test, knn_pred))


# Confusion Matrix for Logistic Regression
print('Confusion Matrix: \n', confusion_matrix(y_test, log_pred))



# Task 3: Summarize Findings in a Report

acc = accuracy_score(y_test, knn_pred)
f1 = f1_score(y_test, knn_pred)


report = f"""

2. Telco Customer Churn Dataset (Classification)
------------------------------------------------
Model Used: K-Nearest Neighbors (KNN)

Accuracy: {acc:.4f}
F1 Score: {f1:.4f}

The KNN model captures customer similarity patterns effectively,
especially after feature scaling.

Conclusion:
-----------
Linear Regression was best suited for housing price prediction,
while KNN provided reasonable churn classification performance.
"""

with open("model_report.txt", "a") as f:
  f.write(report)