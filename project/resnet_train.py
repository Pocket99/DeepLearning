import os
import pydicom
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from skimage import exposure
from PIL import Image
import  gc
import torchvision.utils as vutils
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
gc.collect()
torch.cuda.empty_cache()


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class PneumothoraxDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['ImageId']
        img_path = 'input/' + img_path + '.png'
        #print(img_path)
        img = Image.open(img_path).convert("RGB")

        rle = self.df.iloc[idx]['EncodedPixels']
        label = 0 if rle == '-1' else 1

        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}

from PIL import ImageEnhance, ImageOps, ImageFilter

class RandomBlur:
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img):
        img = Image.fromarray(np.uint8(img))
        radius = torch.rand(1).item() * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
        img = img.filter(ImageFilter.GaussianBlur(radius))
        return np.array(img)


def hist_equalize(img):
    img = np.array(img) # Convert PIL image to numpy array
    img = exposure.equalize_adapthist(img/np.max(img)) # Apply histogram equalization
    img = (img * 255).astype(np.uint8) # Convert the image back to uint8 format
    return img

def adapthist_equalize(img):
    img = np.array(img) # Convert PIL image to numpy array
    img = exposure.equalize_adapthist(img/np.max(img)) # Apply histogram equalization
    img = (img * 255).astype(np.uint8) # Convert the image back to uint8 format
    return img

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: adapthist_equalize(x)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: adapthist_equalize(x)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: adapthist_equalize(x)),
    transforms.ToTensor(),
])


# Load the CSV file
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')


# Create train and validation datasets and data loaders
train_dataset = PneumothoraxDataset(train_df, transform=train_transforms)
val_dataset = PneumothoraxDataset(val_df, transform=val_transforms)
test_dataset = PneumothoraxDataset(test_df, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
# CE
# num_features = model.classifier[6].in_features 
# model.classifier[6] = nn.Linear(num_features, 2)
# BCE
#model.classifier[-1] = nn.Linear(in_features=4096, out_features=1, bias=True)
#model.classifier[-2] = nn.ReLU(inplace=False) # Add ReLU activation after the new linear layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

print(model)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
#criterion = FocalLoss(2)
#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer,'max' ,factor=0.1, patience=3)

writer = SummaryWriter('experiments/Resnet101_1')
#writer = SummaryWriter('runs/test')

print("Starting training...")
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        #print(outputs) # 64, 2
        #print(labels) # 64
        labels = labels.view(-1, 1).float()  # Reshape label tensor to have shape (64, 1)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    scheduler.step(running_loss)
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pred_probs = []
    true_labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            labels = labels.view(-1, 1).float()  # Reshape label tensor to have shape (64, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted_probs = torch.sigmoid(outputs) # Convert to probability values
            pred_probs.extend(predicted_probs.cpu().detach().numpy())
            true_labels.extend(labels.cpu().detach().numpy())

            #_, predicted = torch.max(outputs.data, 1)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    # Calculate ROC/AUC
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    print(f"Val AUC: {roc_auc:.4f}")
    # Write to TensorBoard
    writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)

# Save the model
torch.save(model.state_dict(), f"models/resnet/resnet101_1.pth")

# Test loop
model.eval()
test_running_loss = 0.0
correct = 0
total = 0
pred_probs = []
true_labels = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        images, labels = data["image"].to(device), data["label"].to(device)
        labels = labels.view(-1, 1).float()  # Reshape label tensor to have shape (64, 1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_running_loss += loss.item()
        
        predicted_probs = torch.sigmoid(outputs) # Convert to probability values
        pred_probs.extend(predicted_probs.cpu().detach().numpy())
        true_labels.extend(labels.cpu().detach().numpy())

        # Calculate accuracy
        #_, predicted = torch.max(outputs.data, 1)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_running_loss / len(test_loader)
test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Calculate ROC/AUC
fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
roc_auc = roc_auc_score(true_labels, pred_probs)
print(f"Val AUC: {roc_auc:.4f}")

# Plot ROC curve

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()