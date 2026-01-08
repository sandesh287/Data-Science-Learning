# K-Means Clustering
# K-Means Clustering is an unsupervised learning algorithm that partitions data into k clusters. Each cluster is defined by its centroid, and each data point is assigned to the nearest cluster. The algorithm iteratively adjusts centroids to minimize the variance within each cluster.




# libraries
from sklearn.cluster import KMeans   # KMeans is a clustering algorithm that partitions data into a specified number of clusters (k) by iteratively updating centroids until the clusters stabilize.
import numpy as np



# Sample (features) data (eg. points in 2D space)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])   # Each data point has two values x and y coordinates in 2D space. The shape of x is (6, 2) with two features each.



# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=2, random_state=42)   # this initializes the K-means clustering model with parameters: n_clusters=2, which specifies the data should be partitioned into 2 clusters and random_state=42, sets a random seed for reproducibility, ensuring the initial positions of the centroids are the same each time the code is run.

kmeans.fit(X)   # fits the K-means model to the data X. During this process, the algorithm assigns data points to clusters by iteratively updating the positions of the centroids until convergence. Each point is assigned to the nearest centroid.




# Get the cluster centers (centroids) and labels
centroids = kmeans.cluster_centers_   # this retrieves the coordinates of the cluster centers (centroids), after training. Each centroid represents the mean position of the points within a cluster in 2D space.

labels = kmeans.labels_   # this retrieves the labels assigned to each data point in X, where each label represents the cluster 0 or 1 to which the point belongs. The labels allow you to see which points are grouped together.


print(f'Cluster Centers: \n{centroids}')
print(f'Labels: {labels}')





# This code demonstrates how to use K-means clustering to group data points in 2D space into two clusters. After fitting the model, it outputs the centroids of each cluster and the assigned cluster labels for each data point showing which points belong to which cluster.