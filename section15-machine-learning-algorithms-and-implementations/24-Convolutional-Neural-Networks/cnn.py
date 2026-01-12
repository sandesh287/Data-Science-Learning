# Convolutional Neural Networks (CNNs)
# Convolution Neural Networks (CNNs) are deep learning models specifically designed for precessing structured grid data, such as images. CNNs use convolutional layers that applyu filters to the input image, capturing spatial hierarchies and features like edges, texture, and shapes. CNNs are widely used in computer vision tasks like image classification, object detection, and segmentation.




# libraries
import tensorflow as tf
from keras import layers, models   # These modules are high level API for defining neural network layers and models.
from keras.datasets import mnist   # contains images of handwritten digits from 0 to 9



# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()   # loads MNIST dataset splitting into training and testing sets
X_train, X_test = X_train / 255.0, X_test / 255.0   # Normalize pixel values to a range of 0 to 1, improving training process by scaling pixel intensity from 0 to 1
X_train = X_train.reshape(-1, 28, 28, 1)   # reshape X_train to have shape of number of samples. This reshaping prepares the data for the CNN. In this case, "28x28" is the image size and "1" represents the grayscale channel.
X_test = X_test.reshape(-1, 28, 28, 1)   # reshape X_test data



# Define the CNN model
model = models.Sequential([   # creates a sequential model where each layer is stacked in a particular sequence
  # This adds a 2D convolutional layer. Here, "32" is the number of filters which are feature maps in the layer, (3, 3) is the size of the kernel that slides over the images to detect features, and then activation='relu', it applies the ReLU activation function to introduce non-linearity, input_shape=(28,28,1), this specifies the input shape for the first layer.
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  # This adds a max pooling layer that reduces spatial dimensions by taking the maximum value in each (2x2) window, helping to downsample the feature maps and reduce computation.
  layers.MaxPooling2D((2, 2)),
  # This adds another convolutional layer with 64 filters, a (3x3) kernel and ReLU activation.
  layers.Conv2D(64, (3, 3), activation='relu'),
  # This adds another max pooling layer to further reduce spatial dimension.
  layers.MaxPooling2D((2, 2)),
  # This adds a third convolutional layer with 64 filters and ReLU activation.
  layers.Conv2D(64, (3, 3), activation='relu'),
  # This flattens that 2D matrix from the last convolutional layer into a 1D vector, preparing it for the fully connected layers.
  layers.Flatten(),
  # This fully connected dense layer with 64 neurons and ReLU activation.
  layers.Dense(64, activation='relu'),
  # This is the final output layer with 10 neurons, one for each digit class, which is 0 to 9, and softmax activation, which converts output into probabilities.
  layers.Dense(10, activation='softmax')   # 10-classes for digit 0-9
])



# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # It configures the model for training. "optimizer='adam'", uses the Adam optimizer which combines the benefits of adaptive learning rates and momentum. The "loss='sparse_categorical_crossentropy'", sets the loss function to sparse categorical cross entropy, which is suitable for multi-class classification with the integer labels. "metrics=['accuracy']", tracks the accuracy during training and evaluation.



# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)   # This function trains the model on the training data "X_train, y_train"; "epochs=5", the model will go through the entire training set 5 times; "batch_size=64", updates the model after every 64 samples, improving computational efficiency; "validation_split=0.2", reserves 20% of the training data for validation, allowing the model to evaluate performance on unseen data during training.



# Evalute the model
test_loss, test_acc = model.evaluate(X_test, y_test)   # This evaluates the model on the test data and returns the loss and accuracy. "X_test, y_test" are the test dataset and labels. "test_loss" is the final loss on the test set, and "test_acc" is the accuracy achieved by the model on the test set.

print(f'Test Accuracy: {test_acc}')   # prints test accuracy indicating how well the model generalizes to unseen data






# So, using the dataset MNIST, we trained this model and created a model that can read handwritten numbers and tell us what number it is.