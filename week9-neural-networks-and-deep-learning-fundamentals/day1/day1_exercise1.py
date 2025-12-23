# Introduction to Deep Learning and Neural Networks
# Familiarize yourself with common datasets in deep learning and set up an environment to work with TensorFlow or PyTorch
# Step 1: Explore Common Datasets
  # MNIST: Handwritten digit dataset (28x28 grayscale images, 10 classes)
  # CIFAR-10: 60000  32x32 color images across 10 classes
  # ImageNet: Large dataset for image classification with millions of labeled images
# Step 2: Setup a deep learning environment
  

# As Python 3.14 doesn't support tensorflow, I created a virtual environment with Python 3.11 (latest version of Python that supports tensorflow). Then, I install tensorflow within the virtual environment using 'pip install tensorflow' and now I am working inside virtual environment



# For the warnings
import os
import warnings

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress Python warnings
warnings.filterwarnings("ignore")



# importing libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Verifying if Tensorflow is using GPU (if CUDA is False, it is not using GPU)
# print("TensorFlow version:", tf.__version__)
# print("Built with CUDA:", tf.test.is_built_with_cuda())
# print("GPUs:", tf.config.list_physical_devices('GPU'))


# Load MNIST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f'MNIST Dataset: Train - {X_train_mnist.shape}, Test - {X_test_mnist.shape}')


# Load CIFAR-10
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
print(f'CIFAR-10 Dataset: Train - {X_train_cifar.shape}, Test - {X_test_cifar.shape}')



# Define a basic dense layer
layer = tf.keras.layers.Dense(units=10, activation='relu')
print(f'Tensorflow Layer: {layer}')



# Define a basic dense layer in pytorch
layer_dense = nn.Linear(in_features=10, out_features=5)
print(f'PyTorch Layer: {layer_dense}')



# Visualize MNIST sample
plt.imshow(X_train_mnist[0], cmap='gray')
plt.title(f'MNIST Label: {y_test_mnist[0]}')
plt.show()


# Visualize CIFAR-10 sample
plt.imshow(X_train_cifar[0])
plt.title(f'CIFAR-10 Label: {y_train_cifar[0]}')
plt.show()