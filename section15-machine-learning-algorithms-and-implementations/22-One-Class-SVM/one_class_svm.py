# One-Class SVM (Support Vector Machine)
# One-Class SVM (SUpport Vector Machine) is an algorithm for anomaly detection that identifies data points that differ significantly from the normal distribution of data. It is particularly useful when the dataset primarily consists of one class, and we want to detect outliers. One-Class SVM separates the data into a high-density region (normal data) and sparse regions (anomalies).




# libraries
from sklearn.svm import OneClassSVM   # particularly suited for identifying outliers or anomalies in a dataset
import numpy as np



# Sample data (normal data points clustered around 0)
X = 0.3 * np.random.randn(100, 2)   # creates set of (100,2) dimensional random points drawn from normal distribution, scaled by 0.3. The points are clustered around (0, 0)\
X_train = np.r_[X + 2, X - 2]   # creates two clusters for training by shifting points in X. (X + 2) shifts all points by "+2" in both dimensions, resulting in points clustered around (2, 2) and (X - 2) shifts all point by "-2" in both dimensions, resulting in points clustered around (-2, -2). This concatenates these two clusters vertically to form X_train.



# New test data including some outliers
X_test = np.r_[X + 2, X - 2, np.random.uniform(low=-6, high=6, size=(20, 2))]   # creates test data by combining (X + 2, X - 2)clusters, same as the training data and adding 20 random points as outliers. "np.random.uniform(low=-6, high=6, size=(20, 2))": This generates the (20, 2) dimensional points uniformly distributed between -6 and 6, which are likely outside the main clusters



# Initialize and train the model
model = OneClassSVM(gamma='auto', nu=0.1)   # initializes the one class SVM model with "gamma=auto", sets the kernel coefficient automatically based on the number of features. It controls the influence of each data point, with higher values leading to smaller decision regions, and "nu=0.1", represents upper bound on the fraction of training errors which are anomalies, and the lower bound on the fraction of support vectors. Here it assumes up to 10% of data might be anomalies.

model.fit(X_train)   # trains the model on the X_train data



# Predict on test data (-1 indicates an anomaly, 1 indicates normal)
predictions = model.predict(X_test)   # uses the trained model to predict on X_test. The one class SVM model assigns "-1" to points classified as anomalies and "+1" to normal points



# Display predictions
print(f'Predictions: {predictions}')   # prints the results where each element in prediction is either 1 which is normal or -1 which is  anomaly, based on the model's classification