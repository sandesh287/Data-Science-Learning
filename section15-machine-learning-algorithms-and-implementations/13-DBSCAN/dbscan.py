# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# DBSCAN is an unsupervised clustering algorithm that groups data points based on density, making it particularly effective for identifying clusters of arbitrary shapes and for handling noise (outliers). DBSCAN requires two parameters: eps (the maximum distance between two points to be considered neighbors) and min_samples (the minimum number of points required to form a dense region).




# libraries
from sklearn.cluster import DBSCAN   # DBscan is a density based clustering algorithm that groups data points that are close to each other based on distance and density, and labels points that don't belong to any cluster as noise.
import numpy as np



# Sample Data (eg. points in 2D space)
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])   # Each data point has two values x and y coordinates in 2D space. The shape of x is (6, 2) with two features each.



# Initialize and fit the model
dbscan = DBSCAN(eps=3, min_samples=2)   # initializes the DBSCAN clustering model with parameters: eps=3, which specifies the maximum distance between the two samples for them to be considered as in the same neighborhood. The distance defines the reachability of points. min_samples=2, specifies the minimum number of points required in a neigborhood for a point to be considered a core point. If a point has at least "min_samples" points within its "eps" radius, it forms a cluster, otherwise it may be labeled as noise if it doesn't belong to any cluster.

dbscan.fit(X)   # DBSCAN forms clusters based on the density of points where each point is classified as either a core point, a border point, or noise. The algorithm does not require specifying the number of clusters in advance, as it finds clusters based on the density.



# Get the labels (-1 indicates noise)
labels = dbscan.labels_   # This retrieves the labels assigned to each data point in X after fitting the model. Each label represents the cluster to which the point belongs. Points that are part of a cluster are assigned to a positive integer label, points that are considered noise (they don't belong to the cluster), are assigned to label of "-1".

print(f'Labels: {labels}')






# Output: Labels: [ 0  0  0  1  1 -1] :  These are the labels that are created for the input data, the last input point, which is outside of the numbers that we have here, is considered as noise and others are considered as one cluster. And the ones are considered as cluster one.





# This code demonstrates how to use DBscan to cluster points in 2D space. DBscan is useful for identifying clusters or arbitrary shape and handling noise, which is represented by points with the labels "-1". The algorithm groups points that have a high density and identifies sparse, isolated points as noise.