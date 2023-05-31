import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (MultiplicativeNoise,HorizontalFlip,OpticalDistortion,VerticalFlip,GridDistortion,RandomBrightnessContrast,OneOf,ElasticTransform,RandomGamma,IAAEmboss,Blur,RandomRotate90,Transpose, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import albumentations as A
warnings.filterwarnings("ignore")
# # Read the images
# image1 = cv2.imread('/home/ziruiqiu/comp691_DL/project/experiments/Resnet101_1/cam_pneumothorax_1.jpg')
# image2 = cv2.imread('/home/ziruiqiu/comp691_DL/project/experiments/Resnet101_1/cam_pneumothorax_2.jpg')

# # # Resize the images (optional, for visualization purposes)
# # image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
# # image2 = cv2.resize(image2, (0, 0), fx=0.5, fy=0.5)

# # Concatenate the images horizontally
# composite_image = np.concatenate((image1, image2), axis=1)

# # Display the composite image
# cv2.imshow('Multiple Images', composite_image)

# # Wait for a key press and close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Test Unet

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle

class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

sample_submission_path = '/home/ziruiqiu/comp691_DL/project/data/stage_2_sample_submission.csv'
test_data_folder = "/home/ziruiqiu/comp691_DL/project/data/converted_images"
size = 512
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 8
batch_size = 16
best_threshold = 0.5
min_size = 3500

df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, size, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensorV2(),
            ]
)
path = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.4237.1517875181.859833.png'
image = cv2.imread(path)
image = transform(image=image)["image"]
device = torch.device("cuda:0")
model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
model.eval()
state = torch.load('./model.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])
model.to(device)
preds = torch.sigmoid(model(image.unsqueeze(0).to(device)))
print(preds.shape)
preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
for probability in preds:
    if probability.shape != (1024, 1024):
        probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    predict, num_predict = post_process(probability, best_threshold, min_size)
    if num_predict == 0:
        print('Healthy')
    else:
        print(predict.shape)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024,1024))
        mask = predict
        img[predict==1,0] = 255
        cv2.imwrite('/home/ziruiqiu/comp691_DL/project/random/'+path[-17:-4]+'.png', img)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Close the window
# encoded_pixels = []
# for i, batch in enumerate(testset):
#     preds = torch.sigmoid(model(batch.to(device)))
#     preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
#     for probability in preds:
#         if probability.shape != (1024, 1024):
#             probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
#         predict, num_predict = post_process(probability, best_threshold, min_size)
#         if num_predict == 0:
#             encoded_pixels.append('-1')
#         else:
#             r = run_length_encode(predict)
#             encoded_pixels.append(r)
# df['EncodedPixels'] = encoded_pixels
# df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)