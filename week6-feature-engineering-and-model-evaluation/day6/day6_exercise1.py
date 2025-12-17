# 1. Classification Model Evaluation
# Objective: Train a classification model, calculate confusion matrix, and interpret precision, recall, and F1 score
# Dataset: Iris dataset, focusing on binary classification task (eg. classifying one species with other)



# libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# load dataset
data = load_iris()


# Extracting feature and target
X = data.data
y = (data.target == 0).astype(int)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# predict
y_pred = model.predict(X_test)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Class 0', 'Class 0'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# Classification Metrics
print('\nClassification Report:\n')
print(classification_report(y_test, y_pred))