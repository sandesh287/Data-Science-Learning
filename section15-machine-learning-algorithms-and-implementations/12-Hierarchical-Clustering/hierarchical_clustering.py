# Hierarchical Clustering
# Hierarchical Clustering is an unsupervised learning algorithm that builds a hierarchy of clusters. It starts with each data point as its own cluster and then merges or splits clusters based on distance measures, forming a tree-like structure called a dendrogram. The hierarchy can be used to choose a suitable number of clusters by "cutting" the tree at a specific level.




# libraries
from scipy.cluster.hierarchy import dendrogram, linkage   # linkage performs hierarchical or agglomerative clustering, whereas dendrogram generates a dendrogram plot to visualize the hierarchical clustering.
import matplotlib.pyplot as plt   # provides functions for plotting graphs and visualizations. It is commonly used for visualizing data in Python.
import numpy as np



# Sample Data (eg. points in 2D space)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])   # Each data point has two values x and y coordinates in 2D space. The shape of x is (6, 2) with two features each.



# Perform hierarchical/agglomerative clustering
Z = linkage(X, method='ward')   # ward minimizes the variance within clusters. This function performs hierarchical clustering on X and returns the hierarchical clustering result in the variable Z. X here is the data point to be clustered and method='ward' specifies the linkage criterion, ward is used to minimize the variance within clusters at each step, resulting in a compact and spherical clusters. Other linkage methods include: 'single', 'complete', 'average', which calculates distance between clusters in different ways. Z stores the results of the clustering process as an array, where each row represents a merge containing information on the clusters that were merged, the distance between them, and the number of original data points in the newly formed clusters.



# Plot the dendrogram to visualize the hierarchical clustering
plt.figure(figsize=(8,4))   # This creates a new figure for the plot, with a specified size of 8 by 4 inches.
dendrogram(Z)   # This plots the dendrogram using the hierarchical clustering results Z. The dendrogram visually represents the merging process of clusters, where each U-shaped link shows the distance at which clusters were merged. The x-axis displays the individual data points, and the y-axis shows the distance between merged clusters.

plt.title('Hierarchical Clustering Dendrogram')   # This sets the title of the plot to 'Hierarchical Clustering Dendrogram'.
plt.xlabel('Data Points')   # This sets the label for the x-axis to 'Data Points'.
plt.ylabel('Distance')   # This sets the label for the y-axis to 'Distance'.
plt.show()   # This displays the dendrogram plot, visualizing the hierarchical clustering process. The height of each merge y-axis indicates the distance between clusters, and cutting the dendrogram at different levels can yield different number of clusters.







# This code performs hierarchical clustering on a set of 2D points and visualize the results with the dendogram, which shows how clusters are formed by merging data points and group step by step. The dendrogram allows you to identify a suitable number of clusters by selecting a threshold distance.