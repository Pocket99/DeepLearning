import os
import pydicom
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import torchvision.models as models
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

from PIL import Image
import  gc
import torchvision.utils as vutils

gc.collect()
torch.cuda.empty_cache()

from PneumothoraxDataset import PneumothoraxDataset, train_transforms, val_transforms, test_transforms


# Load the CSV file
df = pd.read_csv('data/stage_2_train.csv')

# Split the data into train and validation sets
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create train and validation datasets and data loaders
train_dataset = PneumothoraxDataset(train_df, transform=train_transforms)
val_dataset = PneumothoraxDataset(val_df, transform=val_transforms)
test_dataset = PneumothoraxDataset(test_df, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)

model = models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)


writer = SummaryWriter('runs/pneumothorax_experiment_VGG16_7_L2_1e-3_epoch20')

def main():
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs = data['image'].to(device)
                labels = data['label'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Write to TensorBoard
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

    # Save the model
    torch.save(model.state_dict(), f"models/pneumothorax_experiment_VGG16_7_grad_cam_L2_1e-3_epoch20_{epoch + 1}.pth")

    # Test loop
    model.eval()
    test_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images, labels = data["image"].to(device), data["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()