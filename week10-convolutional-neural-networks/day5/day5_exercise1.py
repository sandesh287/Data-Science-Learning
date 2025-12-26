# Building CNN Architectures with PyTorch
# Build, Train, Evaluate and Experiment with CNNs for CIFAR-10 classification using PyTorch



# importing librearies
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


# Define transformations
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f'\nTraining Data Size: {len(train_dataset)}')
print(f'\nTest Data Size: {len(test_dataset)}')


# Define a CNN model
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  
  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x


model = CNN()

print(model)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs=10):
  model.train()
  for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
      # Zero Gradient
      optimizer.zero_grad()
      
      # Forward Pass
      outputs = model(images)
      loss = criterion(outputs, labels)
      
      # Backward Pass and optimize
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
    print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}')


train_model(model, train_loader, criterion, optimizer)


# Evaluate loop
def evaluate_model(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in test_loader:
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print(f'\nTest Accuracy: {100 * correct / total}%')


evaluate_model(model, test_loader)