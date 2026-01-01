# Positional Encoding and Feed-Forward Networks
# Implement positional encoding and integrate it with a basic Transformer model
# Experiment with different positional encoding methods and observe the effects



# libraries
import numpy as np
import matplotlib.pyplot as plt



# Implement positional encoding

# Define Positional Encoding function
def positional_encoding(seq_len, embed_dim):
  pos = np.arange(seq_len)[:, np.newaxis]
  i = np.arange(embed_dim)[np.newaxis, :]
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
  angle_rads = pos * angle_rates
  
  # Apply sine to even indices and cosine to odd indices
  pos_encoding = np.zeros(angle_rads.shape)
  pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])   # even indices
  pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])   # odd indices
  return pos_encoding


# Generate positional encoding
seq_len = 50
embed_dim = 16
pos_encoding = positional_encoding(seq_len, embed_dim)


# Visualize positional encoding
plt.figure(figsize=(10,6))
plt.pcolormesh(pos_encoding, cmap='viridis')
plt.colorbar()
plt.title('Positional Encoding')
plt.xlabel('Embedding dimension')
plt.ylabel('Position')
plt.show()




# Integrate positional encoding with a basic Transformer model

# libraries
import torch
import torch.nn as nn


# Define class for Transformer with positional encoding
class TransformerWithPositionalEncoding(nn.Module):
  def __init__(self, embed_dim, seq_len, num_heads, ff_dim):
    super(TransformerWithPositionalEncoding, self).__init__()
    self.embedding = nn.Embedding(seq_len, embed_dim)
    self.positional_encoding = nn.Parameter(torch.tensor(positional_encoding(seq_len, embed_dim), dtype=torch.float32))
    self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ff_dim),
      nn.ReLU(),
      nn.Linear(ff_dim, embed_dim)
    )
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
  
  # Define forward function
  def forward(self, x):
    # Add positional encoding to embedding
    x = self.embedding(x) + self.positional_encoding
    # self attention
    attn_output, _ = self.multihead_attention(x, x, x)
    x = self.norm1(x + attn_output)
    # Feed forward network
    ffn_output = self.ffn(x)
    x = self.norm2(x + ffn_output)
    return x


# Define model parameters
embed_dim = 16
seq_len = 50
num_heads = 4
ff_dim = 64

model = TransformerWithPositionalEncoding(embed_dim, seq_len, num_heads, ff_dim)

print(model)




# Experiment with different positional encoding methods and observe the effects

# Learnable positional encoding
class LearnablePositionalEncoding(nn.Module):
  def __init__(self, seq_len, embed_dim):
    super(LearnablePositionalEncoding, self).__init__()
    self.positional_encoding = nn.Parameter(torch.zeros(seq_len, embed_dim))
  
  # Forward function
  def forward(self, x):
    return x + self.positional_encoding


learnable_pe_model = LearnablePositionalEncoding(seq_len, embed_dim)

print(learnable_pe_model)