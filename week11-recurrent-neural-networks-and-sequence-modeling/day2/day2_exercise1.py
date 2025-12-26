# Understanding RNN Architecture and Backpropagation Through Time (BPTT)
# Build a simple RNN model for text classification using TensorFlow.
# Train the RNN and observe how it captures sequence patterns



# importing libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# Building hyperparameters
vocab_size = 10000
max_len = 200


# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# Preprocess data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f'\nTraining Data Shape: {X_train.shape}')
print(f'Test Data Shape: {X_test.shape}')


# Build RNN Model
model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  SimpleRNN(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Model Architecture summary
model.summary()


# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f'\nTest Loss: {loss} , Test Accuracy: {accuracy}')