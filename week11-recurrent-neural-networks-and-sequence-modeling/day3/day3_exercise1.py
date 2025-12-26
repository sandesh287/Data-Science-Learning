# Long Short-Term Memory (LSTM) Networks
# Build an LSTM model for sentiment analysis on the IMDB Movie Reviews dataset and compare its performance with a basic RNN model. (using TensorFlow)



# libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense


# Define Hyperparameters
vocab_size = 10000
max_len = 200


# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# Preprocess data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f'\nTraining Data Shape: {X_train.shape}')
print(f'Test Data Shape: {X_test.shape}')


# Build basic RNN model
rnn_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  SimpleRNN(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid')
])


# Compile RNN model and view summary
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()


# Train RNN model
rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


# Evaluate RNN model
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)

print(f'RNN Test Loss: {rnn_loss} , RNN Test Accuracy: {rnn_accuracy}')



# Build and Train LSTM model

# Build LSTM model
lstm_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  LSTM(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid')
])


# Compile and view summary of LSTM model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()


# Train LSTM model
lstm_history = lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


# Evaluate LSTM model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)

print(f'LSTM Test Loss: {lstm_loss} , LSTM Test Accuracy: {lstm_accuracy}')



# print('From the output, we can see that LSTM accuracy is 85% and accuracy for simple RNN model is 65%. So, LSTM is far better than the simple RNN model.')