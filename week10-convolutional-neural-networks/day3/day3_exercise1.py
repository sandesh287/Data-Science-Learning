# Pooling Layers and Dimensionality Reduction
# Implement max pooling and average pooling layers on feature maps and observe their effects on size and representation.



# libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, uniform_filter   # maximum_filter for max pooling and uniform_filter for average pooling


# Create a sample feature map
feature_map = np.array([
  [1, 2, 3, 0],
  [4, 5, 6, 1],
  [7, 8, 9, 2],
  [0, 1, 2, 3]
])
# Defines a 2D array feature map as a sample feature map (4x4)


# Max Pooling (2x2)
max_pooled = maximum_filter(feature_map, size=2, mode='constant')
# Performing max pooling with kernel size (2x2), each region in feature map is replaced with its maximum value


# Average Pooling (2x2)
avg_pooled = uniform_filter(feature_map, size=2, mode='constant')
# Performing average pooling with kernel size (2x2), each region in feature map is replaced with its average value


# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(feature_map, cmap='viridis')
axes[0].set_title('Original Feature Map')
axes[1].imshow(max_pooled, cmap='viridis')
axes[1].set_title('Max Pooled')
axes[2].imshow(avg_pooled, cmap='viridis')
axes[2].set_title('Average Pooled')
plt.show()



# Implement Pooling Layer in Tensorflow

# libraries
import tensorflow as tf


# Create a sample input tensor (1x4x4x1) <- batch_size, height, width, channels
input_tensor = tf.constant(feature_map.reshape(1, 4, 4, 1), dtype=tf.float32)
# Converting the feature map into a 4D tensor with dimensions of batch_size, height, width, channels resp.


# Max Pooling
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
# Defining (2x2) max pooling layer with strides of 2

max_pooled_tensor = max_pool(input_tensor)
# applied max pooling to the input tensor


# Average Pooling
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')
# Defining (2x2) average pooling layer with strides of 2

avg_pooled_tensor = avg_pool(input_tensor)
# applied average pooling to the input tensor

print(f'\nMax Pooled Tensor:\n {tf.squeeze(max_pooled_tensor).numpy()}')
print(f'\nAverage Pooled Tensor:\n {tf.squeeze(avg_pooled_tensor).numpy()}')
print('\n\n\n')



# Implement Pooling Layer in PyTorch

# libraries
import torch
import torch.nn as nn


# Create a sample input tensor (batch_size, channels, height, width)
input_torch = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# converting the feature map to a 4D tensor with dimensions of batch size, channels, height and width. And then unsqueeze it twice here.


# Max Pooling
max_pool_torch = nn.MaxPool2d(kernel_size=2, stride=2)

max_pooled_torch = max_pool_torch(input_torch)


# Average Pooling
avg_pool_torch = nn.AvgPool2d(kernel_size=2, stride=2)

avg_pooled_torch = avg_pool_torch(input_torch)

print(f'\nMax Pooled Torch:\n {max_pooled_torch.squeeze().numpy()}')
print(f'\nAverage Pooled Torch:\n {avg_pooled_torch.squeeze().numpy()}')





# Combining Convolution and Pooling Layers

# Tensorflow example
model_tf = tf.keras.Sequential([
  tf.keras.Input(shape=(32, 32, 3)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.AveragePooling2D((2, 2))
])



# PyTorch Example
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.pool2 = nn.AvgPool2d(2, 2)
  
  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = self.pool1(x)
    x = torch.relu(self.conv2(x))
    x = self.pool2(x)
    return x

