# Set up a transfer learning environment, load a pre-trained model, and explore its architecture and layers
# Using PyTorch



# libraries
import torch
import torchvision.models as models


# load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)


# Print model architecture
print(model)


# Freeze the model parameters
for param in model.parameters():
  param.requires_grad = False


# Modify the final layer for new task
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)

print('Modified Model: \n', model)


# Unfreeze specific layers
for name, param in model.named_parameters():
  if 'layer4' in name:
    param.requires_grad = True


# Access specific layers
for name, param in model.named_parameters():
  print(f'Layer-{name}, Required Grad: {param.requires_grad}')