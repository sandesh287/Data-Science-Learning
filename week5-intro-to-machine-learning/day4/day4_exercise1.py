# I. Implement a logistic regression model to classify a dataset (eg. predicting if a customer will make a purchase)
# Objective: Train a logistic regression model to classify customers as likely to purchase which is 1 or 0
# Dataset: Social Network ads dataset
# Steps: 1. Load and preprocess the dataset
# 2. Split the data into training and testing set
# 3. Train a logistic regression model
# 4. Evaluating the model using classification metrics
# Will learn how to train Logistic Regression model to classify customers as likely to purchase or not, based on their behaviour data which is Age and Salary.

# II. Analyze model Performance
# Visualizing the decision boundary and analyzing the model predictions



# I. Implement a Logistic Regression model to classify a dataset.
# libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Generate synthetic dataset
np.random.seed(42)

n_samples = 200
X = np.random.rand(n_samples, 2) * 10
y = (X[:, 0] * 1.5 + X[:, 1] > 15).astype(int)


# Create a dataframe
df = pd.DataFrame(X, columns=['Age', 'Salary'])
df['Purchase'] = y


# Split data
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Salary']], df['Purchase'], test_size=0.2, random_state=42)


# Train the Logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate performance
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred))
print('\nClassification Report: \n', classification_report(y_test, y_pred))



# II. Analyze the model performance

# libraries
import matplotlib.pyplot as plt


# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


# Predict probabilities for grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Plot
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolor='k', cmap='coolwarm')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()