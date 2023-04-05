import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from PneumothoraxDataset import PneumothoraxDataset, test_transforms
from torch.autograd import Variable
import numpy as np
import cv2
from torch.nn import functional as F
import os

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.gradients = []

        for module in self.model.named_modules():
            if isinstance(module[1], nn.ReLU):
                module[1].register_forward_hook(self.save_feature_maps)
                module[1].register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps.append(output.detach())

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, input_tensor, class_index=None):
        self.feature_maps = []
        self.gradients = []

        output = self.model(input_tensor)
        print("output: ", output)
        if class_index is None:
            class_index = torch.argmax(output).item()
        print("class_index: ", class_index)
        output[0][class_index].backward()

        feature_map = self.feature_maps[-1]
        grads = self.gradients[-1]
        weights = torch.mean(grads, dim=[2, 3], keepdim=True)
        cam = torch.mul(feature_map, weights).sum(dim=1, keepdim=True)
        cam = nn.ReLU()(cam)
        cam = cam.cpu().numpy()[0, 0]

        return cam

# read the image
img_path = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.3677.1517875178.954801.png'
img1 = Image.open(img_path)
img1 = img1.convert('RGB')
print("img.shape: ", img1.size)
#img1.show()
img = test_transforms(img1)
print("img.shape: ", img.shape)
img = img.unsqueeze(0)
img = Variable(img, requires_grad=True).cuda()


#load the model
device = torch.device("cuda")
model = models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)
model.load_state_dict(torch.load('models/pneumothorax_experiment_VGG16_7_grad_cam_L2_1e-3_epoch20_20.pth'))
model.to(device)
model.eval()
print(model)
# Apply Grad-CAM
grad_cam = GradCAM(model)
cam = grad_cam(img)

# Resize and normalize the Grad-CAM heatmap
cam = cv2.resize(cam, img1.size)
cam = cam - np.min(cam)
cam = cam / np.max(cam)

# Overlay the Grad-CAM heatmap onto the original image
overlay = np.uint8(255 * cam)
heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
#heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)# Convert color channel order from BGR to RGB

img1_np = np.array(img1)
result = cv2.addWeighted(img1_np, 1.0, heatmap, 0.6, 0)
# Save and show the result
cv2.imwrite('grad_cam_result.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
Image.fromarray(result).show()