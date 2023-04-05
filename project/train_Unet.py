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
from PIL import Image
# Functions to encode and decode RLE masks
def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

# def rle2mask(rle, width, height):
#     mask= np.zeros(width* height)
#     array = np.asarray([int(x) for x in rle.split()])
#     starts = array[0::2]
#     lengths = array[1::2]

#     current_position = 0
#     for index, start in enumerate(starts):
#         current_position += start
#         mask[current_position:current_position+lengths[index]] = 255
#         current_position += lengths[index]

#     return mask.reshape(width, height)

def rle2mask(rle, shape):
    height, width = shape
    if rle == '-1':
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros(height * width, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(height, width).T

# Custom dataset class
class PneumothoraxDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ImageId']
        rle = row['EncodedPixels']

        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = Image.open(img_path)
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)

        mask = rle2mask(rle, (image.shape[0], image.shape[1]))
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return {'image': image, 'mask': mask}

# Function to set up transformations (resize and normalize)
def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.9, border_mode=0),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])

def criterion(outputs, masks, model, lambda_l1=1e-5):
    bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
    l1_loss = sum(p.abs().sum() for p in model.parameters())
    return bce_loss + lambda_l1 * l1_loss

# Load the CSV file
df = pd.read_csv('data/stage_2_train.csv')

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create train and validation datasets and data loaders
train_dataset = PneumothoraxDataset(train_df, 'input/', transform=get_transforms())
val_dataset = PneumothoraxDataset(val_df, 'input/', transform=get_transforms())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. resnet34, resnet50, efficientnet-b0, etc.
    encoder_weights="imagenet",     # use pre-trained weights for encoder initialization
    in_channels=1,                  # input channels (DICOM images have 1 channel)
    classes=1,                      # output channels (binary mask)
    activation=None            # use sigmoid activation for binary segmentation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter('runs/pneumothorax_experiment')

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs = data['image'].to(device)
        masks = data['mask'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        if len(masks.size()) == 4 and masks.size(-1) == 1:
            masks = masks.permute(0, 3, 1, 2).float()  # Reorder the dimensions to match the model outputs
        
        loss = criterion(outputs, masks, model)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs = data['image'].to(device)
            masks = data['mask'].to(device)

            outputs = model(inputs)
            if len(masks.size()) == 4 and masks.size(-1) == 1:
                masks = masks.permute(0, 3, 1, 2).float()  # Reorder the dimensions to match the model outputs
            loss = criterion(outputs, masks, model)
            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Write to TensorBoard
    writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
