# Introduction to Attention Mechanisms
# Implement a basic Attention Mechanism using NumPy and visualize its impact on a simple sequence task.



# libraries
import numpy as np



# Using NumPy

# Define queries, keys and values
queries = np.array([[1,0,1],[0,1,1]])
keys = np.array([[1,0,1],[1,1,0],[0,1,1]])
values = np.array([[10,0],[0,10],[5,5]])


# Compute Attention Scores
scores = np.dot(queries, keys.T)


# Apply softmax to normalize scores
def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)


# Compute weighted sum of values
context = np.dot(attention_weights, values)

print(f'\nAttention Weights: \n {attention_weights}')
print(f'Context Vector: \n {context}')