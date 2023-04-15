from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import torch.nn as nn
from PIL import Image
from skimage import exposure
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
#pthfile = 'models/pneumothorax_experiment_VGG16_8_grad_cam_L2_1e-3_epoch20_20.pth'
pthfile = '/home/ziruiqiu/comp691_DL/project/models/resnet/resnet18_1.pth' #BCE
model.load_state_dict(torch.load(pthfile))
model.eval()														# 8
target_layers = [model.layer4[-1]]

def adapthist_equalize(img):
    img = np.array(img) # Convert PIL image to numpy array
    img = exposure.equalize_adapthist(img/np.max(img)) # Apply histogram equalization
    img = (img * 255).astype(np.uint8) # Convert the image back to uint8 format
    return img

# def img_preprocess(img_in):
#     img = img_in.copy()						
#     img = img[::-1, :, :]   				# 1
#     img = np.ascontiguousarray(img)			# 2
#     #print("img.shape: ", img.shape)
#     cv2.resize(img, (224, 224))
#     transform = transforms.Compose([
#         transforms.Lambda(lambda x: adapthist_equalize(x)),
#         transforms.ToTensor(),
#         #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img = transform(img)
#     img = img.unsqueeze(0)					# 3
#     return img
# # Note: input_tensor can be a batch tensor with several images!



# # 图片读取；网络加载
# img = cv2.imread(path_img, 1)[:, :, ::-1]
# img = cv2.resize(img, (224, 224))
# transform = transforms.Compose([
#         transforms.Lambda(lambda x: adapthist_equalize(x)),
#         transforms.ToTensor(),
#         #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# img = transform(img)
# img = img.unsqueeze(0)
# img = np.float32(img) / 255
# input_tensor = torch.from_numpy(img)

# print("input_tensor.shape: ", input_tensor.shape)

#path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.4200.1517875181.692066.png' # 0
path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.3678.1517875178.953520.png' # pneumothorax

img = cv2.imread(path_img, 1)[:, :, ::-1]
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
transform = transforms.Compose([
        transforms.Lambda(lambda x: adapthist_equalize(x)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img1 = transform(img)
img1 = img1.unsqueeze(0)
input_tensor = img1
print("input_tensor.shape: ", input_tensor.shape)
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

targets = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=False)
cv2.imwrite('/home/ziruiqiu/comp691_DL/project/experiments/Resnet18_1/pytorch_pneumothorax.jpg', visualization)