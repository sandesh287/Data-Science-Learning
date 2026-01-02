# 1. Load a pre-trained ResNet or VGG model and fine-tue it for a new image classification task (eg. classifying animals or plants).
# 2. Experiment with freezing and unfreezing layers and observe the impact on performance.

# Using PyTorch



# libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# Load Pre-trained model
model = models.resnet50(pretrained=True)


# Freeze all layers
for param in model.parameters():
  param.requires_grad = False


# Replace the last layer for a new task
num_features = model.fc.in_features
model.fc = nn.Sequential(
  nn.Linear(num_features, 256),
  nn.ReLU(),
  nn.Dropout(0.4),
  nn.Linear(256, 5),
  nn.Softmax(dim=1)
)

print(model)


# Data preparation
transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('PATH_TO_FOLDER_TRAIN', transform=transform)   # replace with your training data path
val_data = datasets.ImageFolder('PATH_TO_FOLDER_VAL', transform=transform)   # replace with your validation data path

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
for epoch in range(10):
  model.train()
  for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
  
  print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# Unfreeze specific layers (Layer 4)
for name, param in model.named_parameters():
  if 'layer4' in name:
    param.requires_grad = True


# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for inputs, labels in val_loader:
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total}%')