# t-Distributed Stochastic Neighbor Embedding (t-SNE)
# t-SNE is a dimensionality reduction technique primarily used for visualizing high-dimensional data in 2D or 3D space. Unlike PCA< t-SNE is non-linear and focuses on preserving the local structure of data, making it highly effective for visualizing clusters. However, it is computationally intensive and best suited for small to medium-sized datasets.




# libraries
from sklearn.manifold import TSNE   # t-SNE is a nonlinear dimensionality reduction technique, particularly used for visualizing high dimensional data in 2D or 3D by preserving the local structure of the data points.
import numpy as np



# Sample data (eg. points in high-dimensional space)
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7], [5, 7, 8], [8, 9, 10], [9, 10, 11]])   # defines the feature data X as a numpy array where each sublist represents a data point in a 3D space with three values x, y, and z coordinates. Shape of X is (7, 3), where there are 7 data points with 3 features each.



# Initialize and fit the model
tsne = TSNE(n_components=2, perplexity=5, random_state=42)   # reducing to 2D for visualization. initialize t-SNE model with parameters: "n_components=2", which specifies target number of dimensions for the reduced data. Setting n_components=2, reduces the data from original 3D form into 2D, making it easier to visualize. "perplexity=5", controls the balance between local and global aspects of the data in the embedding. It's usually recommended to set the value between 5 and 50. Here, setting perplexity to 5 should work with seven data points. However, if your dataset grows, you may increase the perplexity to optimize the t-SNE output.

X_reduced = tsne.fit_transform(X)   # fits the t-SNE model to the data X and transform it into 2D space. The "fit_transform()" both fits the model and applies transformation, generating X_reduced, a new array containing 2D coordinates for each data point. Unlike PCA, t-SNE does not capture variance, but instead focuses on preserving local structure, so points that were close in high dimensional space will remain close in the reduced 2D space.


print(f'Reduced Data: \n {X_reduced}')





# This code demonstrate how to use t-SNE to reduce a dataset from 3D to 2D for visualization purposes, or it is particularly effective at creating visually interpretable representations of complex high dimensional data by clustering similar points close together, revealing patterns that may not be apparent in higher dimensions.