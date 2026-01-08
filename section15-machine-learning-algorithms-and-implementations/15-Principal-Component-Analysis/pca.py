# Principal Component Analysis (PCA)
# Principal Component Analysis (PCA) is a dimensioanlity reduction technique used to transform a high-dimensional dataset into a lower-dimensional one, by identifying the directions (principal components) that capture the maximum variance in the data. PCA is widely used for data visualization, noise reduction, and speeding up machine learning algorithms by reducing the number of features.




# libraries
from sklearn.decomposition import PCA   # PCA is dimensionality reduction technique that transforms data into a lower dimensional space by identifying the directions, which are principal components that capture the most variance in the data.
import numpy as np



# Sample data (eg. points in a 3D space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8]])   # each sublist represents a data point in 3D space with three values x, y, and z coordinates. For example, (1, 2, 3), and (5, 6, 7) are two points in 3D space. The shape of X is (5, 3), meaning there are five data points with three features each.



# Initialize and fit the model
pca = PCA(n_components=2)   # Reducing to 2 dimensions; initializes PCA model with parameter: n_components=2, which specifies the number of principal components (which are dimensions to keep). And n_components=2 means we are reducing the data from 3D to a 2D object.

X_reduced = pca.fit_transform(X)   # fits the PCA model to the data X and transform it into a lower dimensional space. The "fit_transform()" both fits the model, finding the principal components and transforms the data based on these components, resulting in "X_reduced" which has a shape of (5, 2).


print(f'Reduced Data:\n {X_reduced}')   # each row in "X_reduced" represents a data point in 2D space, where each value represents the projection of the original data onto the first and the second Principal components.

print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')   # indicates how much of the total variance in the original data is captured by each principal component. "explained_variance_ratio_" is an array where each value represents the proportion of variance explained by a principal component. This information is helpful for understanding how much information is retained in the reduced dimensional representation.






# This code demonstrates how to use PCA to reduce the 3D data set to a 2D, while retaining as much variance as possible. The "explained_variance_ratio_" shows the contribution of each principal component to the total variance, providing insight into the effectiveness of the dimensionality reduction.