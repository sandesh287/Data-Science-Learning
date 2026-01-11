# Isolation Forest
# Isolation Forest is an ensemble method for anomaly detection that isolates anomalies rather than profiling normal data. The algorithm randomly selects a feature and a split value to partition the data, creating trees where anomalies are easier to isolate due to their sparse distribution. Anomalies are identified based on their shorter path lengths in the tree structure, as they are isolated faster than normal points.




# libraries
from sklearn.ensemble import IsolationForest   # efficient for identifying outliers, by isolating observations through random partitioning
import numpy as np



# Sample data (normal data points clustered around 0)
X = 0.3 * np.random.randn(100, 2)   # generates set of (100,2) dimensional random points drawn from normal distribution, scaled by 0.3. This scaling reduces the spread, creating the points clustered around (0, 0)
X_train = np.r_[X + 2, X - 2]   # creates a dataset with points around two clusters. creates two clusters for training by shifting points in X. (X + 2) shifts all points by "+2" in both dimensions, resulting in points clustered around (2, 2) and (X - 2) shifts all point by "-2" in both dimensions, resulting in points clustered around (-2, -2). This concatenates these two clusters vertically to form X_train.



# New test data including some outliers
X_test = np.r_[X + 2, X - 2, np.random.uniform(low=-6, high=6, size=(20, 2))]   # creates test data by combining (X + 2, X - 2)clusters, same as the training data and adding 20 random points as outliers. "np.random.uniform(low=-6, high=6, size=(20, 2))": This generates the (20, 2) dimensional points uniformly distributed between -6 and 6, which are likely outside the main clusters



# Initialize and train the model
model = IsolationForest(contamination=0.1, random_state=42)   # contamination is the expected proportion of outliers. This initializes the isolation forest model with "contamination=0.1", specifies the proportion of data expected to be outliers. Setting it to 0.1 means the model assumes about 10% of the data points could be anomalies. "random_state=42", sets a random seed for reproducibility, ensuring that results remain consistent across runs

model.fit(X_train)   # trains the isolation forest model on X_train data



# Predict on test data (-1 indicates an anomaly, 1 indicates normal)
predictions = model.predict(X_test)   # uses the trained model to predict on X_test. The isolation forest model assigns "-1" to points classified as anomalies and "+1" to normal points



# Display predictions
print(f'Predictions: \n {predictions}')   #  prints the results where each element in prediction is either 1 which is normal or -1 which is anomaly, based on the model's classification. This allows us to see which points in X_test the model flagged as outliers.