# Text Preparation and Word Embeddings for RNNs
# Preprocess a text dataset and integrate word embeddings (eg. GloVe) into an LSTM model for sentiment analysis
# We will be using GloVe

# downloaded the dataset from: https://nlp.stanford.edu/projects/glove/   and used glove.6B.100d.txt file



# libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt


# Hyperparameters
vocab_size = 10000
max_len = 200


# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# Decode reviews to text for preprocessing
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_reviews = [" ".join([reverse_word_index.get(i-3, "?") for i in review]) for review in X_train[:5]]


# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f'\nTraining Data shape: {X_train.shape}, {y_train.shape}')
print(f'Test Data shape: {X_test.shape}, {y_test.shape}')



# Load GloVe Embeddings
embedding_index = {}
glove_file = "glove.6B.100d.txt"
with open(glove_file, "r", encoding='utf-8') as file:
  for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs

print(f'\nLoaded {len(embedding_index)} word vectors.')


# Prepare Embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
  if i < vocab_size:
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector



# Build LSTM model using GloVe embeddings

# Define LSTM model with GloVe Embeddings
glove_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
  LSTM(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid')
])


# Compile the model
glove_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
glove_model.summary()


# Train the model
glove_history = glove_model.fit(
  X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1
)


# Evaluate the model
glove_loss, glove_accuracy = glove_model.evaluate(X_test, y_test, verbose=0)



# Build LSTM model without using GloVe embeddings

# Define LSTM model without GloVe Embeddings
lstm_model = Sequential([
  Embedding(input_dim=vocab_size, output_dim=128),
  LSTM(128, activation='tanh', return_sequences=False),
  Dense(1, activation='sigmoid')
])


# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()


# Train the model
lstm_history = lstm_model.fit(
  X_train, y_train, validation_split=0.2, epochs=5, batch_size=64
)


# Evaluate the model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)

print(f'\nLSTM model with GloVe Embeddings: Test Accuracy: {glove_accuracy} , Test Loss: {glove_loss}')
print(f'\nLSTM model without GloVe Embeddings: Test Accuracy: {lstm_accuracy} , Test Loss: {lstm_loss}')



# Plot accuracy comparison
models = ['LSTM', 'LSTM GloVe']
accuracies = [lstm_accuracy, glove_accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Comarison of Accuracy LSTM with and without word embeddings')
plt.ylabel('Accuracy')
plt.show()

# Plot Loss comparison
models = ['LSTM', 'LSTM GloVe']
losses = [lstm_loss, glove_loss]
plt.bar(models, losses, color=['blue', 'green'])
plt.title('Comarison of Loss LSTM with and without word embeddings')
plt.ylabel('Loss')
plt.show()




print('Conclusion: Here, we can see that Accuracy for LSTM without GloVe is: 85% and with GloVe is: 56%. Similarly, Loss for LSTM without GloVe is: 37% and with GloVe is: 65%. In this particular case, it looks like it added more data than it needed  and kind of made the accuracy much worse as compared to with LSTM, which might happen sometimes depending on dataset. So, it is always good to try with different combinations to see which model and with what technique it works perfectly fine.')