# Apply data augmentation to a dataset and train a fine-tuned model. Experiment with hyperparameters to observe their impact on performance.

# Using PyTorch



# libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# Load pre-tained MobileNetV2
model = models.mobilenet_v2(pretrained=True)


# Freeze all the models
for param in model.parameters():
  param.requires_grad = False


# Replace classification head
model.classifier[1] = nn.Linear(model.last_channel, 5)


# Define Data Augmentation
train_transform = transforms.Compose([
  transforms.RandomRotation(20),
  transforms.RandomHorizontalFlip(),
  transforms.RandomResizedCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
])


# Create training data
train_data = datasets.ImageFolder('TRAINING_IMAGE_FOLDER', transform=train_transform)   # replace with your train dataset path
val_data = datasets.ImageFolder('VALIDATION_IMAGE_FOLDER', transform=val_transform)   # replace with your validation dataset path


# Train and validation loader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


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




# Note: So, once you have trained the model, next step you can tune the hyperparameters. You can change the learning rate, batch sizes, test different optimizers (SGD)