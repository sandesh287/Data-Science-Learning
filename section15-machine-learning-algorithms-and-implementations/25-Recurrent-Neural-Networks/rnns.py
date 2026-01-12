# Recurrent Neural Networks (RNNs)
# Recurrent Neural Networks (RNNs) are neural networks designed for sequential data, such as time series, language, or speech. RNNs have connections that form cycles, allowing them to retain information from previous steps in the sequence. This makes RNNs well-suited for tasks like text generation, language modeling, and time series forecasting. A common variant, Long Short-Term Memory (LSTM), helps to address the issue of long-term dependency.




# libraries
import tensorflow as tf   # main library for deep learning, providing tools to define and train neural networks
from keras import layers, models   # used to define the layers and structure of the neural network
from keras.datasets import imdb   # a dataset of movie reviews, with each review labeled as positive or negative sentiment
from keras.preprocessing import sequence   # A utility for preprocessing sequences specifically to pad or truncate sequences to the same length



# Load and preprocess the IMDB dataset
max_features = 1000   # Vocabulary size, sets the vocabulary size to the top 10,000 most frequent words in the dataset
max_len = 500   # limit reviews to 500 words, reviews longer than 500 words will be truncated, while shorter ones will be padded


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)   # loads the IMDb dataset, limiting it to max features most frequent words. "(X_train, y_train)" are the training data and labels. Each review is represented as a sequence of integers, which are word indices, and "(X_test, y_test)" are test data and labels.
X_train = sequence.pad_sequences(X_train, maxlen=max_len)   # This pads or truncates each review in X_train to ensure all the reviews have exactly max length words
X_test = sequence.pad_sequences(X_test, maxlen=max_len)   # This step ensures that each review is the same length, which is necessary for batch processing in neural networks.



# Define the RNN model
# defines a sequential model where each layer's output is passed to the next layer
model = models.Sequential([
  # The embedding layer converts word indices into dense vectors of fixed size. "max_features", vocabulary size limiting embedding to the top 10,000 words; "32", size of each word vector, which is the embedding dimension; "input_length=max_len", length of the input sequences, which is 500 number of words per review
  layers.Embedding(max_features, 32, input_length=max_len),
  # This is simple RNN layer with 32 units which processes the sequence data.
  layers.SimpleRNN(32),
  # This is the output layer with 1 unit and a sigmoid activation function. Here, "1", parameter represents a single output neuron predicting the probability of a positive or negative sentiment; Sigmoid activation function outputs a probability between 0 and 1 suitable for binary classification.
  layers.Dense(1, activation='sigmoid')
])



# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   # This configures the model for training. "optimizer='adam'", uses the Adam optimizer which adjusts learning rates based on momentum and adaptive learning rates; "loss='binary_crossentropy'", the binary cross entropy loss function is suitable for binary classification task, which is sentiment analysis; "metrics=['accuracy']", which specifies accuracy as the evaluation metric.



# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)   # trains the model on the training data. "(X_train, y_train)", is the training data and labels; "epochs=5", is the number of times the model will go through the entire training dataset; "batch_size=64", which is the number of samples per batch for gradient updates, improving computational efficiency; "validation_split=0.2", reserves 20% of the training data for validation, allowing the model to evaluate performance on unseen data during training.



# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)   # evaluates the model on the test set, returning the test loss and accuracy. The "X_test, y_test", are the test data and label. "test_loss", which is the calculated loss on the test data, and the "test_acc" is the accuracy of the model on the test data.

print(f'Test Accuracy: {test_acc}')