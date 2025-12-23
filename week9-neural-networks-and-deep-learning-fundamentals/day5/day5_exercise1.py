# Building Neural Networks with TensorFlow and Keras
# Build, train, evaluate, and save a simple neural network to classify digits from the MNIST dataset



# tensorflow.keras.model : contains tools for building neural network models
# Sequential : linear stack of layers, it allows you to add one layer at a time, going from input to output sequentially
# tensorflow.keras.layers : provides building blocks for layers in a neural network
# Dense : fully connected layer where each neuron is connected to every neuron in the previous layer
# Flatten : flattens a multi dimensional input into a single vector for fully connected layers
# Conv2D : 2D convolutional layer that applies filters to the input image to extract features
# MaxPooling2D : reduces the spatial dimensions of the input while retaining the important features, used for downsampling
# Dropout : regularization technique to randomly set a fraction of the input units to zero during training, preventing overfitting


# import libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Normalize data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f'Training Data Shape: {X_train.shape}')
print(f'Test Data Shape: {X_test.shape}')



# Conv2D : 
  # 32 : no. of filters; learnable feature detectors
  # (3, 3): filter kernel size meaning each filter is a (3 x 3) matrix.
  # activation='relu' : it applies ReLU activation function, which sets negative values to zero, introducing non-linearity
  # input_shape=(28, 28, 1) : shape of input image which is 28x28 pixels with 1 channel, which is the grayscale
# MaxPooling2D(2, 2) : 2x2 pool size, it reduces each 2x2 block in the input to a single maximum value, reduces the spatial dimensions by half
# Flatten : adds a flatten layer to convert the 2D feature map output of the previous layer into a 1D vector, necessary for connecting convolutional layers to fully connected layers
# Dense(128, activation='relu') : dense layer with 128 neurons, and activation='relu', which applies the ReLU activation function
# Dropout(0.5) : adds a dropout layer, 0.5 is the fraction of neurons to randomly drop during training, helps prevent overfitting
# Dense(10, activation='softmax') : adds the final output layer, 10 neurons is corresponds to the 10 output classes. Activation softmax outputs a probability distribution across those 10 different classes.

# Build the model
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  MaxPooling2D(2, 2),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax')
])


# model.summary() : displays the model architecture showing each layer type, output shape and the number of parameters. Also total number of trainable and non trainable parameters in the model.

# Display Model Architecture
model.summary()


# Compile the model
model.compile(
  optimizer='Adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)


# Train the model
history = model.fit(
  X_train, y_train,
  epochs=10,
  batch_size=32,
  validation_split=0.2
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {test_accuracy}')


# Save the model
model.save('mnist_classifier.h5')


# Load the model
from tensorflow.keras.models import load_model
loaded_model = load_model('mnist_classifier.h5')


# Verify loaded model performance
loss, accuracy = loaded_model.evaluate(X_test, y_test)

print(f'Loaded Model Accuracy: {accuracy}')