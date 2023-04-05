import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import torch.nn as nn
from PIL import Image
# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()						
    img = img[::-1, :, :]   				# 1
    img = np.ascontiguousarray(img)			# 2
    #print("img.shape: ", img.shape)
    cv2.resize(img, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    print("heatmap.shape: ", heatmap.shape)
    cam_img = 0.3 * heatmap + 0.7 * img
    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)

if __name__ == '__main__':
    #path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.3678.1517875178.953520.png' # pneumothorax
    path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.4200.1517875181.692066.png' # 1

    json_path = 'labels.json'
    output_dir = 'cam'

    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value)
               in load_json.items()}
	
	# 只取标签名
    classes = list(classes.get(key) for key in range(2))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)
    #img = Image.open(path_img).convert("RGB")
    img_input = img_preprocess(img)

    # 加载 squeezenet1_1 预训练模型
    net = models.vgg16(pretrained=True)
    num_features = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_features, 2)
    pthfile = 'models/pneumothorax_experiment_VGG16_8_grad_cam_L2_1e-3_epoch20_20.pth'

    net.load_state_dict(torch.load(pthfile))
    net.eval()														# 8
    print(net)

    # 注册hook
    net.features[-1].register_forward_hook(farward_hook)	# 9
    net.features[-1].register_backward_hook(backward_hook)

    # forward
    print("input:", img_input.shape)
    output = net(img_input)
    print("output: {}".format(output))
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = output[0,idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, output_dir)
