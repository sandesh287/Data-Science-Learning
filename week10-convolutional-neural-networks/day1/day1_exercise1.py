# Introduction to Convolutional Neural Networks (CNNs)
# Visualize images in a dataset, explore their pixel data, and set up an environment for building CNNs using TensorFlow or PyTorch.



# As I am using tensorflow in this week as well, so I have created a virtual environment for this week's model as well. The virtual environment is of Python 3.11, which supports the latest versions of TensorFlow and PyTorch.
# In this exercise, I had saved my model in data folder inside the current working directory. But I cannot push it to github, as it is large in size. 



# importing libraries
import matplotlib.pyplot as plt   # creating charts and visualization
from torchvision import datasets, transforms
# datasets provides access to the popular datasets (like CIFAR-10), transforms contains utilities to preprocess and transform image data, such as converting images to tensors
import numpy as np
import torch.nn.functional as F


# Load dataset
transform = transforms.ToTensor()
# defines a transformation to convert images from the data set into PyTorch tensors. This is necessary for using the data in PyTorch models

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# loads the CIFAR-10 dataset; root='./data' : directory where dataset will be stored; train=True : loads the training split of dataset
# CIFAR10 is 10 because it classifies all the images into 10 different categories 0 - 9


# Visualize sample images
fig, axes = plt.subplots(1, 5, figsize=(12,3))
# creates a figure with 1 row and 5 columns of subplots for displaying images

for i in range(5):
  image, label = train_dataset[i]
  axes[i].imshow(image.permute(1, 2, 0))
  # displaying image using imshow() method; permute() method reorders the dimensions from (channels, height, width) to (height, width, channels); (1, 2, 0) : Usually its (0,1,2):(channels,height,width), channels to end, height to first and width to second
  axes[i].axis('off')   # hiding the axis from graph for cleaner display
  axes[i].set_title(f'Label: {label}')   # title of each subplot

plt.show()


# Explore pixel data
# Display pixel values for the first image
image, label = train_dataset[0]   # retrieves the first image and its label from training dataset
print(f'Label: {label}')
print(f'Image Shape: {image.shape}')
print('Pixel Values: ')
print(image)




# Setup an environment to build CNNs using Tensorflow

# importing libraries
import tensorflow as tf   # imports tensorflow library for building and training deep learning models


# Define a simple CNN Model
model = tf.keras.Sequential([   # this is a sequential model, means each layer runs one at a time in sequential order
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  # First layer is Conv2D layer with 32 filters, (3x3) kernels, relu activation for image that we have
  tf.keras.layers.MaxPooling2D((2, 2)),
  # MaxPooling2D layer with (2x2) pooling window to reduce spatial dimensions of previous input
  tf.keras.layers.Flatten(),
  # flattens the 2D feature maps into 1D vector for fully connected layers
  tf.keras.layers.Dense(128, activation='relu'),
  # Fully connected layers with 128 neurons
  tf.keras.layers.Dense(10, activation='softmax')
  # fully connected layers with 10 neurons, for 10 different classification units
])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# compiles the model, configurations configures the model with optimizer of Adam (Adam Optimizer); loss='sparse_categorical_crossentropy' : specifies the loss function for multi-class classification; metrics=['accuracy'] : tracks accuracy during the training

print(f'TensorFlow CNN Model is ready.')




# CNN Model using PyTorch

# libraries
import torch.nn as nn   # provides tools for building neural networks


# Define simple CNN Model
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, activation='relu')
    # creating convolutional layer with 3 input channels, 32 filters, (3x3) kernels
    self.pool = nn.MaxPool2d(2, 2)
    # max pooling layer with (2x2) pooling window
    self.fc1 = nn.Linear(32 * 15 * 15, 128)
    # creating fully connected layer transforming it from (32*15*15) into 128
    self.fc2 = nn.Linear(128, 10)
    # creating fully connected layer, converting input size 128 neurons to output size 10 classes

  def forward(self, x):   # defines forward pass
    x = F.relu(self.conv1(x))  # applies convilution conv1 and relu activation
    x = self.pool(x)   # apply max pooling
    x = x.view(-1, 32 * 15 * 15)   # flatten the tensor
    x = F.relu(self.fc1(x))   # passing data through fully connected layers of fc1
    x = self.fc2(x)   # passing data through fully connected layers of fc2

print(f'PyTorch CNN Model is ready!')