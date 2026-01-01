# Implement a basic Attention Mechanism using PyTorch and visualize its impact on a simple sequence task.



# libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



# Using PyTorch

# Define queries, keys and values
queries = torch.tensor([[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]])
keys = torch.tensor([[1.0, 0.0, 1.0],[1.0, 1.0, 0.0],[0.0, 1.0, 1.0]])
values = torch.tensor([[10.0, 0.0],[0.0, 10.0],[5.0, 5.0]])


# Compute attention scores
scores = torch.matmul(queries, keys.T)


# apply softmax to normalize scores
attention_weights = F.softmax(scores, dim=-1)


# compute weighted sum of values
context = torch.matmul(attention_weights, values)

print(f'\nAttention Weights: \n {attention_weights}')
print(f'Context Vector: \n {context}')


# Visualize attention weights
plt.matshow(attention_weights)
plt.colorbar()
plt.title('Attention Weights using PyTorch')
plt.show()