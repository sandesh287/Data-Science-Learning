# RNN Project: Sentiment Analysis
# Build, train, and optimize RNN, LSTM, and GRU models for Sentiment Analysis



# libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU
import matplotlib.pyplot as plt


# Load dataset
vocab_size = 10000
max_len = 100

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f'\nTraining Data Shape: {X_train.shape}, {y_train.shape}')
print(f'\nTest Data Shape: {X_test.shape}, {y_test.shape}')



# Simple RNN model

# Define RNN model
rnn_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  SimpleRNN(128, activation='tanh'),
  Dense(1, activation='sigmoid')
])


# Compile the model
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Display model summary
rnn_model.summary()



# LSTM Model

# Define LSTM model
lstm_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  LSTM(128, activation='tanh'),
  Dense(1, activation='sigmoid')
])


# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Display model summary
lstm_model.summary()



# GRU Model

# Define GRU model
gru_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  GRU(128, activation='tanh'),
  Dense(1, activation='sigmoid')
])


# Compile the model
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Display model summary
gru_model.summary()



# Train models

# Train RNN model
history_rnn = rnn_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Train LSTM model
history_lstm = lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Train GRU model
history_gru = gru_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)



# Evaluate models

# Evaluate RNN model
loss_rnn, accuracy_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)

# Evaluate LSTM model
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)

# Evaluate GRU model
loss_gru, accuracy_gru = gru_model.evaluate(X_test, y_test, verbose=0)


print(f'\nRNN Test Accuracy: {accuracy_rnn} , RNN Test Loss: {loss_rnn}')
print(f'LSTM Test Accuracy: {accuracy_lstm} , LSTM Test Loss: {loss_lstm}')
print(f'GRU Test Accuracy: {accuracy_gru} , GRU Test Loss: {loss_gru}')



# Plot

# Plot training accuracy
plt.plot(history_rnn.history['accuracy'], label='RNN Training Accuracy')
plt.plot(history_lstm.history['accuracy'], label='LSTM Training Accuracy')
plt.plot(history_gru.history['accuracy'], label='GRU Training Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot training loss
plt.plot(history_rnn.history['loss'], label='RNN Training Loss')
plt.plot(history_lstm.history['loss'], label='LSTM Training Loss')
plt.plot(history_gru.history['loss'], label='GRU Training Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




print('Hence, we can see that, accuracy of RNN is: 60%, LSTM is: 83.74% and GRU is: 83.24%. So, we can conclude that, in this particular dataset, LSTM model is better than other two. And also we can see that the difference in accuracy of LSTM and GRU is so small. So, the accuracy also depends on dataset as well. Or maybe if we increase our epochs, the GRU might be better.')