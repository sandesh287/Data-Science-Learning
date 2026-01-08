# Gradient Boosting
# Gradient Boosting is an ensemble technique that builds a series of decision trees, where each tree corrects the errors of the precious ones. By combining the predictions of these trees, Gradient Boosting models create a more accurate final prediction. Popular implementations include XGBoost, LightGBM, and CatBoost, which are optimized for speed and accuracy.




# libraries
from sklearn.ensemble import GradientBoostingClassifier   # ensemble method that builds multiple weak learners (typically decision trees) sequentially, where each tree tries to connect the errors of the previous ones.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np



# Sample data (eg. hours studied and grades vs. pass/fail)
X = np.array([[1,50], [2,60], [3,55], [4,65], [5,70], [6,75], [7,80], [8,85], [9,90], [10,95]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # 0 = fail, 1 = pass. This is binary classification task, ideal for gradient boosting model.



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)   # initializes gradient boosting classifier model with parameters: n_estimators=100 specifies the number of boosting stages (trees) in the ensemble. A higher value generally improves accuracy, but increases computation time. learning_rate=0.1 controls the contribution of each tree to the final prediction. A smaller learning rate requires more trees, but it can lead to better generalization. random_state=42 ensures reproducibility by setting a random seed.

model.fit(X_train, y_train)



# Make predictions
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



# Display results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n {conf_matrix}')






# This code demonstrates the use of gradient boosting classifier to predict whether a student will pass or fail based on the hours studied and grades. The model's performance is evaluated using accuracy and confusion matrix, providing a comprehensive view of its classification results.