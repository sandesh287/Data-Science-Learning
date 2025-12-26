# Introduction to Sequence Modeling and RNNs
# Preprocess a text dataset for use in RNNs and set up an environment in TensorFlow for building RNNs.



# importing libraries
import tensorflow as tf   # for deep learning operations
from tensorflow.keras.datasets import imdb
# imdb dataset is a standard dataset for sentiment analysis, where each review is represented as a sequence of integers
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pad_sequences function ensure all sequences which are reviews, are of the same length by padding or truncating them
from tensorflow.keras.models import Sequential
# sequential class for building a sequential, which is layer-by-layer model.
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
# Embedding converts word indices to dense vectors. SimpleRNN implements a basic RNN layer and Dense is fully connected layer for output


vocab_size = 10000   # vocabulary size : top 10000 most frequent words will be used here
max_len = 200   # maximum sequence length : Reviews will be truncated or padded to 200 words, so every review is 200 words


# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# Preprocess data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
# ensuring that all the reviews have the same length, which is max length of 200. padding='post' : padding happens at the end and truncates reviews longer than 200 words, if it's more than 200 words.

X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f'\nTraining Data Shape: {X_train.shape}')
print(f'\nTest Data Shape: {X_test.shape}')


# Build the model
model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  SimpleRNN(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid'),
])
# using Sequential(), which is a model built for stacking layers sequentially, 
# with Embedding() setting (input_dim=vocab_size) where input vocabulary size is 10,000 words and (output_dim=128), where each word index is mapped to a 128 dimensional vector; 
# initializing simple RNN where number of RNN units to be used is 128. Activation function is tanh, (return_sequences=False): outputs only the last time steps results
# Dense(): creating a fully connected output layer. 1 : outputs single value for binary classification. (activation=sigmoid) ensures the output is between 0 and 1 probability


# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# (optimizer='adam'): optimer for training loss; (loss='binary_crossentropy'): loss function for binary classification and metrics; (metrics=['accuracy']): evaluates the model's accuracy during the training process


# Model architecture summary, including layers, parameters and output shapes
model.summary()


# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# (X_train, y_train): training data and labels; (epochs=5): train for five iterations; (batch_size=32): processes 32 samples at a time; (validation_split=0.2): uses 20% of the training data for validation


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
# Tests the model on unseen test dataset, output will be final loss on test set, accuracy will be accuracy on test set

print(f'\nTest Loss: {loss} , Test Accuracy: {accuracy}')