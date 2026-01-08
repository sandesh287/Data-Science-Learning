# Gaussian Mixture Models (GMM)
# Gaussian Mixture Models (GMM) is a probabilistic clustering algorithm that assumes data points are generated from a mixture of several Gaussian distributions with unknown parameters. GMM assigns a probability to each data point for belonging to each cluster, making it a soft clustering technique. It is particularly useful when clusters have different shapes or densities.




# libraries
from sklearn.mixture import GaussianMixture   # clustering model that assumes data is generated from a mixture of multiple Gaussian distributions. It is used for identifying clusters in data when clusters may have elliptical shapes.
import numpy as np



# Sample data (eg. points in 2D space)
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])   # Each data point has two values x and y coordinates in 2D space. The shape of x is (6, 2) with two features each.



# Initialize and fit the model
gmm = GaussianMixture(n_components=2, random_state=42)   # initializes a Gaussian Mixture Model with parameters: n_components=2, which specifies the number of Gaussian distributions or clusters to fit to the data. This parameter represents the assumed number of clusters in the data. random_state=42, sets a random seed for reproducibility, ensuring the initialization of the model is same each time the code runs.

gmm.fit(X)   # During this process, the model estimates the parameters of the Gaussian distribution mean and co-variance that best describe each cluster in the data.



# Get the cluster labels and probabilities
labels = gmm.predict(X)   # predicts the cluster labels for each data points in X. The predict method assigns each data point to the cluster with the highest probability. The output is an array of integer "labels" representing the cluster to which each point belongs.
probs = gmm.predict_proba(X)   # calculates the probability that each data point belongs to each cluster. The predict_proba() method outputs an array where each row represents a data point and each column represents a cluster. The values indicate the probability that each point belongs to each cluster, providing a soft clustering assignment i.e., allowing each point to have a probability distribution over clusters.


print(f'Cluster Labels: {labels}')
print(f'Cluster Probabilities:\n {probs}')





# This code demonstrates how to use a Gaussian mixture model for clustering points in 2D space. Unlike K-means, which assigns hard cluster labels, GMM provides soft cluster assignments by assigning each point a probability distribution over clusters, making it particularly useful for overlapping or elliptical clusters.