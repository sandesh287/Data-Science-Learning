# Convolutional Layers and Filters
# Understand convolution operations by implementing and visualizing their effects using TensorFlow and PyTorch



# importing libraries
import numpy as np   # for numerical computation and handling arrays
import matplotlib.pyplot as plt   # creating visualization like plots or images
from scipy.ndimage import convolve   # convolve : to perform convolution operation on images


# Load a sample grayscale image
image = np.random.rand(10, 10)
# generates a (10x10) random grayscale image, values between 0 and 1 to simulate an example of an image

print(image)


# Define convolution kernels (filters)
edge_detection_kernel = np.array([
  [-1, -1, -1],
  [-1, 8, -1],
  [-1, -1, -1]
])
# (3x3) kernel filter used for detecting edges in the image by emphasizing high contrast areas

blur_kernel = np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
]) / 9
# (3x3) kernel used for blurring the image by Averaging the pixel values in a neighborhood; / 9 : It normalizes the kernel so that the sum of all the elements equals one, ensuring the brightness remains consistent.


# Apply convolution
edge_detected_image = convolve(image, edge_detection_kernel)
blurred_image = convolve(image, blur_kernel)
# applies the convolution operation on the image using the specified kernel


# Visualize the original and filtered image
fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(edge_detected_image, cmap='gray')
axes[1].set_title('Edge Detected Image')
axes[2].imshow(blurred_image, cmap='gray')
axes[2].set_title('Blurred Image')
plt.show()



# Implement Convolution in tensorflow

# libraries
import tensorflow as tf


# Create a sample input tensor (batch_size, height, width, channels) <-  in order
image_tensor = tf.random.normal([1, 10, 10, 1])   # creates a random (10x10) grayscale image tensor, with batch_size of 1 (first 1) and channel of 1 (last 1)


# Define a convolution layer
conv_layer = tf.keras.layers.Conv2D(
  filters=1,
  kernel_size=(3,3),
  strides=(1,1),
  padding='same'
)
# defining a 2D convolutional layer in TensorFlow; filters=1 : specifies the number of output channels or filters; kernel_size=(3,3) : defines the size of the convolutional kernel; strides=(1,1) : specifies the step size of the convolution; padding='same' : ensuring the output size matches the input size by padding the borders


# Applying convolution
output_tensor = conv_layer(image_tensor)   # applies the convolution to the image tensor

print(f'\nOriginal Shape (Tensor): {image_tensor.shape}')
print(f'Output Shape (Tensor): {output_tensor.shape}')



# Implement Convolution in PyTorch

# Libraries
import torch
import torch.nn as nn


# Create a sample input tensor (batch_size, channels, height, width)  <-  in order
image_torch = torch.randn([1, 1, 10, 10])


# Define a convolutional layer
conv_layer_torch = nn.Conv2d(
  in_channels=1,
  out_channels=1,
  kernel_size=3,
  stride=1,
  padding=1
)


# Apply convolution
output_torch = conv_layer_torch(image_torch)

print(f'\nOriginal Shape (Torch): {image_torch.shape}')
print(f'Output Shape (Torch): {output_torch.shape}')




# Tensorflow Example after increasing kernel size from (3x3) to (5x5)
conv_layer_large_kernel = tf.keras.layers.Conv2D(filters=1, kernel_size=(5,5), strides=(1,1), padding='same')
output_large_kernel = conv_layer_large_kernel(image_tensor)

print(f'\nLarge Kernel Output Shape (Tensor): {output_large_kernel.shape}')




# PyTorch example after increasing stride from 1 to 2
conv_layer_stride_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
output_layer_stride_2 = conv_layer_stride_2(image_torch)

print(f'\nStripe Output Shape (Torch): {output_layer_stride_2.shape}')