# Preprocess a text dataset for use in RNNs and set up an environment in PyTorch for building RNNs.


# importing libraries
import torch   # core PyTorch library for tensor operations
import torch.nn as nn   # module for building neural networks
import torch.optim as optim   # optimizer for training neural network
from torch.utils.data import DataLoader, TensorDataset    # utilities for working with dataset which includes DataLoader for batching and TensorDataset for packaging data.
from tensorflow.keras.datasets import imdb   # import IMDB datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences   # for preprocessing


# Building Hyperparameters
vocab_size = 10000
max_len = 200


# Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# Preprocess data
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')


# Prepare data for PyTorch
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
# converts numpy arrays into PyTorch tensors and creates a dataset of paired inputs and labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# batches the dataset into groups of 32 and shuffles the data for training


# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(RNNModel, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    # init function defines model architecture; (nn.Embedding): embeds the input word indices into dense vectors; (nn.RNN): simple recurrent neural network layer; (nn.Linear): fully connected layer mapping RNN outputs to the final binary sentiment prediction
    
  # Forward function
  def forward(self, x):
    embedded = self.embedding(x)
    output, hidden = self.rnn(embedded)
    return torch.sigmoid(self.fc(hidden.squeeze(0)))
  # embed the input sequences; passes embedded sequences through RNN; hidden contains the RNNs last hidden state; applies fully connected layer to hidden state; uses sigmoid activation to output the probabilities


# Initialize the model
model = RNNModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, output_dim=1)
# initializing RNN model; vocabulary size 10000; embedding dimension 128; hidden dimension 128; output dimension is one for binary classification, so it will be either 0 or 1.


# Define Loss and Optimizers
criterion = nn.BCELoss()   # BCELoss() : Binary Cross-Entropy Loss
# computes the binary cross-entropy loss for the predictions

optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizes the model using Adam with a learning rate of 0.001


# Train the model
def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
  model.train()
  for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      predictions = model(X_batch).squeeze(1)
      loss = criterion(predictions, y_batch.float())
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      # model.train(): sets the model to training mode; 
      # For each epoch in range of epochs: 
        # initialize cumulative loss (epoch_loss = 0) 
        # For each batch in the train_loader: 
          # (optimizer.zero_grad()): clears the previous gradients that has been calculated; (predictions): makes the predictions here; (loss = criterion()): calculates the loss; (loss.backward()): computing the gradients;  (optimizer.step()): updates the model weights; (epoch_loss += loss.item()): adding batch loss to epoch loss and keep on incrementing
    print(f'Epoch: {epoch+1} , Loss: {epoch_loss / len(train_loader)}')


train_rnn(model, train_loader, criterion, optimizer)


# Evaluation Loop
def evaluate_rnn(model, X_test, y_test):
  model.eval()   # set the model to evaluation mode
  with torch.no_grad():   # disables gradient computation for efficiency
    predictions = model(torch.tensor(X_test)).squeeze(1)
    loss = criterion(predictions, torch.tensor(y_test).float())
    accuracy = ((predictions > 0.5) == torch.tensor(y_test).float()).float().mean().item()
  print(f'Test Loss: {loss.item()} , Test Accuracy: {accuracy}')


evaluate_rnn(model, X_test, y_test)