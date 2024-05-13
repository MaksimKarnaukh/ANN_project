from dataset import load_data
from models import EfficientNetModel
from helper_functions import train

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
pretext_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

scene_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Define datasets
pretext_dataset = datasets.ImageFolder(root="./data/15SceneData/train", transform=pretext_transform)
scene_dataset = datasets.ImageFolder(root="./data/15SceneData/train", transform=scene_transform)

# Define data loaders
pretext_loader = DataLoader(pretext_dataset, batch_size=32, shuffle=True, num_workers=4)
scene_loader = DataLoader(scene_dataset, batch_size=32, shuffle=True, num_workers=4)


# Define pretext task classifier
class PretextTaskClassifier(nn.Module):
    def __init__(self):
        super(PretextTaskClassifier, self).__init__()
        self.model = efficientnet_b0(pretrained=True)  # Load pre-trained EfficientNet-B0
        # Modify classifier for pretext task (Gaussian blur kernel size prediction)
        self.model.classifier = nn.Linear(1280, 5)  # 5 classes for different kernel sizes

    def forward(self, x):
        return self.model(x)


# Define scene classification model
class SceneClassificationModel(nn.Module):
    def __init__(self, num_classes=15):
        super(SceneClassificationModel, self).__init__()
        self.model = efficientnet_b0(pretrained=True)  # Load pre-trained EfficientNet-B0
        # Modify classifier for scene classification task
        self.model.classifier = nn.Linear(1280, num_classes)  # 15 classes for scene classification

    def forward(self, x):
        return self.model(x)


# Training function for pretext task
def train_pretext_task(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# Initialize pretext task classifier
pretext_model = PretextTaskClassifier().to(device)

# Define pretext task training parameters
pretext_criterion = nn.CrossEntropyLoss()
pretext_optimizer = optim.Adam(pretext_model.parameters(), lr=0.001)

# Train pretext task classifier
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_pretext_task(pretext_model, pretext_criterion, pretext_optimizer, pretext_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Pretext Task Loss: {train_loss:.4f}")

# Save pretext task classifier
torch.save(pretext_model.state_dict(), "pretext_model.pth")

# Initialize scene classification model
scene_model = SceneClassificationModel().to(device)

# Load weights from the pretext task classifier
pretext_model.load_state_dict(torch.load("pretext_model.pth"))

# Freeze layers in scene classification model except the classifier
for param in scene_model.model.parameters():
    param.requires_grad = False

# Define scene classification training parameters
scene_criterion = nn.CrossEntropyLoss()
scene_optimizer = optim.Adam(scene_model.parameters(), lr=0.001)


# Training function for scene classification task
def train_scene_classification(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# Train scene classification model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_scene_classification(scene_model, scene_criterion, scene_optimizer, scene_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Scene Classification Loss: {train_loss:.4f}")

# Save scene classification model
torch.save(scene_model.state_dict(), "scene_model.pth")