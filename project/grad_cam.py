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
output_dir = 'cam/'
# 将要可视化的图片输进网络模型，判断出所属类别
# 获取最后一个卷积层的输出特征图
# 通过图片所属类别，得到权重，对获取的特征图的各个通道赋值，并且相加为单通道的特征图
df = pd.read_csv('data/stage_2_train.csv')
test_dataset = PneumothoraxDataset(df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)

# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W = img.size
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    print(heatmap.shape)
    # img_temp = img_temp.squeeze(0)
    # img_temp = np.transpose(img_temp, (1, 2, 0))
    # print(img_temp.shape)
    #img.show()
    img = np.asarray(img) / 255
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)


# read the image
img_path = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.4200.1517875181.692066.png'
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
finalconv_name = 'features'  # name of the last convolutional layer
model.eval()
print(model)

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

 # 存放梯度和特征图
fmap_block = list()
grad_block = list()

# 注册hook
model.features[-1].register_forward_hook(farward_hook)
model.features[-1].register_backward_hook(backward_hook)
print(model.features[-1])
# forward
output = model(img)
idx = np.argmax(output.cpu().data.numpy())
print(idx)

# backward
model.zero_grad()
loss = output[0][idx]
loss.backward()

# 生成cam
grads_val = grad_block[0].cpu().data.numpy().squeeze()
fmap = fmap_block[0].cpu().data.numpy().squeeze()

# 保存cam图片
cam_show_img(img1, fmap, grads_val, output_dir)














# # 定义计算CAM的函数
# def returnCAM(feature_conv, weight_softmax, class_idx):
#     # 类激活图上采样到 256 x 256
#     size_upsample = (256, 256)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     print(weight_softmax[class_idx].shape)	#
#     # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
#     # 				feature_conv.shape为(1, 512, 7, 7)
#     # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512)
#     # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)
#     cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
#     print(cam.shape)		# 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
#     cam = cam.reshape(h, w) # 得到单张特征图
#     # 特征图上所有元素归一化到 0-1
#     cam_img = (cam - cam.min()) / (cam.max() - cam.min())  
#     # 再将元素更改到　0-255
#     cam_img = np.uint8(255 * cam_img)
#     output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam
# # get the convolutional feature map
# features_blobs = []     # 后面用于存放特征图

# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())

# # 获取 features 模块的输出
# model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# # get the softmax weight
# net_name = []
# params = []
# for name, param in model.named_parameters():
#     net_name.append(name)
#     params.append(param)
# print(net_name[-1], net_name[-2])	        # classifier.6.bias classifier.6.weight
# print(len(params))		                    # 32
# logit = model(img)			                # 计算输入图片通过网络后的输出值
# print(logit.shape)						    # torch.Size([1, 2])
# print(params[-2].data.cpu().numpy().shape)	# (2, 4096)
# print(features_blobs[0].shape)              # (1, 512, 7, 7)



# weight_softmax = np.squeeze(params[-2].data.cpu().numpy())	
# print(weight_softmax.shape)					# (2, 4096)
# h_x = F.softmax(logit, dim=1).data.squeeze()	
# print(h_x.shape)						# 
# probs, idx = h_x.sort(0, True)
# probs = probs.cpu().numpy()					# 概率值排序
# idx = idx.cpu().numpy()						# 类别索引排序，概率值越高，索引越靠前
# print(probs, idx)

# # 对概率最高的类别产生类激活图
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
# # 融合类激活图和原始图片
# img = cv2.imread(img_path)
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.7
# cv2.imwrite('CAM0.jpg', result)
