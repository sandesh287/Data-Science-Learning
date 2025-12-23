# Building Neural Networks with PyTorch
# Build, train, evaluate, and save a neural network for MNIST digit classification using PyTorch



# torchvision.datasets : provides popular datasets like MNIST, CIFAR-10
# torchvision.transforms : provide transformations for pre-processing data, example converting to tensor and normalizing
# DataLoader : utility to handle data batching shuffling and loading in parallel
# torch.nn : provides classes and functions for building and defining neural networks. It contains predefined building blocks like layers, loss functions, activation functions and more.
# torch.nn.functional : contains functions for operations that can be applied directly without defining them as layers. These are stateless functions, meaning they don't have parameters that need to be learned like weights or biases. And it's commonly used for applying activation functions or loss computation directly.


# importing libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# transforms.Compose : combines a list of transformations to be applied sequentially
# transforms.ToTensor() : converting image data into PyTorch tensors which scales pixel values to 0 to 1
# transforms.Normalize((0.5,),(0.5,)) : normalizing tensor data to have a mean of 0.5 and a standard deviation of 0.5 for better training stability


# Define transformation
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,),(0.5,))
])


# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True) : 
  # This is for training dataset
  # download=True : downloading the MNIST dataset if not already present
  # root='./data' : directory to store the data set. So it will create a folder called as data and save it there.
  # train=True : specifies whether to load the training or testing set
  # transform=transform : applies the defined transformations to the dataset


# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# DataLoader() : loads data in batches for training and testing
  # batch_size=32 : number of samples per batch
  # shuffle=True : shuffles the training data to ensure randomness.
  # shuffle=False : keeps the test data in order for evaluation


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(f'Training Data Size: {len(train_dataset)}')
print(f'Test Data Size: {len(test_dataset)}')


# nn.module : base class for all PyTorch models
# nn.Flatten() : flattens the 2D image 28x28 into 1D vector
# self.fc1 = nn.Linear(28 * 28, 128) : First layer; input size is 784 which is the 1D vector from my flatten. So flatten that into a size of 128.
# self.fc2 = nn.Linear(128, 64) : Second layer; input that comes from this 128, changing that to an output size of 64
# self.fc3 = nn.Linear(64, 10) : Third layer; output size in 10, for 10 different classes that we have for numbers


# Define the model
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(28 * 28, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)
  
  def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


# Initialize model
model = NeuralNetwork()
print(model)


# criterion : loss function using cross entropy for multi-class classification
# optimizer : because we have 10 different outputs here. Specifying Adam optimizer for updating model weights and learning rate to 0.001


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
  model.train()   # puts the model in training mode
  for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
      # Zero Gradients
      optimizer.zero_grad()
      
      # Forward Pass
      outputs = model(images)
      loss = criterion(outputs, labels)
      
      # Backward pass and optimize
      loss.backward()
      optimizer.step()   # update the model parameters
      
      running_loss += loss.item()
    print(f'Epoch: {epoch}, Loss: {running_loss / len(train_loader)}')


train_model(model, train_loader, criterion, optimizer)


# Evaluate Loop
def evaluate_model(model, test_loader):
  model.eval()   # puts the model in evaluation mode
  correct = 0
  total = 0
  with torch.no_grad():   # disables gradient computation for efficiency
    for images, labels in test_loader:
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)   # get the class with the highest probability
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print(f'Test Accuracy: {100 * correct / total}%')


evaluate_model(model, test_loader)


# Save model
torch.save(model.state_dict(), 'mnist_model.pth')


# Load the model
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('mnist_model.pth'))

# Verify loaded model performance
evaluate_model(loaded_model, test_loader)


# # Update optimizer with new learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# train_model(model, train_loader, criterion, optimizer)
# evaluate_model(model, test_loader)