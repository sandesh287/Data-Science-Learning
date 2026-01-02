# Introduction to Transfer Learning
# Set up a transfer learning environment, load a pre-trained model, and explore its architecture and layers
# Using TensorFlow



# libraries
import tensorflow as tf
from keras.applications import ResNet50


# load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')


# Display model architecture
model.summary()


# Access specific layers
for i, layer in enumerate(model.layers):
  print(f'Layer {i}: {layer.name}, Trainable: {layer.trainable}')


# Freeze all the layers except the top 10
for layer in model.layers[:-10]:
  layer.tainable = False