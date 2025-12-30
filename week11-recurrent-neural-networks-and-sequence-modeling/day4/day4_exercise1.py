# Gated Recurrent Unit (GRUs)
# Build a GRU-based model for the IMDB Movie Reviews Dataset and compare its performance with the LSTM model



# libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense


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



# RNN Model
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



# Build and Train GRU model

# Build GRU model
gru_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  GRU(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid')
])


# Compile and view summary of GRU model
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.summary()


# Train GRU model
gru_history = gru_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


# Evaluate GRU model
gru_loss, gru_accuracy = gru_model.evaluate(X_test, y_test)

print(f'GRU Test Loss: {gru_loss} , GRU Test Accuracy: {gru_accuracy}')



print('From the output, we can see that LSTM accuracy is 85.93%, with loss of 37%, GRU accuracy is 86.5%, with loss of 46%, and accuracy for simple RNN model is 54%, with loss of 68%.')