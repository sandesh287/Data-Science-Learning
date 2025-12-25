# Building CNN Architectures with Keras and TensorFlow
# Build, Train, and Evaluate a CNN for image classification on the MNIST or CIFAR-10 dataset using Keras and TensorFlow.



# importing libraries
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical   # utility to convert integer labels to one hot encoded labels
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt



# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Loads dataset and split into training and testing set; X_train and X_test contains image data and y_train and y-test contains the corresponding labels for those images


# Normalize data
X_train = X_train.astype('float32') / 255.0
# Converts the image data to 32-bit floating point number for compatibility with tensorflow models, and divide by 255.0 means, it normalizes the pixel values from 0 to 255 to 0 to 1 to improve model convergence during training process

X_test = X_test.astype('float32') / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Converts the integer labels 0-9 to one hot encoding format

print(f'\nTraining Data Shape: {X_train.shape}, Label Shapes: {y_train.shape}')
print(f'\nTest Data Shape: {X_test.shape}, Label Shapes: {y_test.shape}')



# Build CNN Model
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax')
])


# Display model summary
model.summary()   # display model architecture


# Compile the model
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)


# Train the model
history = model.fit(
  X_train, y_train,
  epochs=10,
  batch_size=64,
  validation_split=0.2
)


# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'\nTest Accuracy: {test_accuracy}')


# Visualize the training process

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()