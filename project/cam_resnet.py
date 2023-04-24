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


# 图片预处理
def adapthist_equalize(img):
    img = np.array(img) # Convert PIL image to numpy array
    img = exposure.equalize_adapthist(img/np.max(img)) # Apply histogram equalization
    img = (img * 255).astype(np.uint8) # Convert the image back to uint8 format
    return img

def img_preprocess(img_in):
    img = img_in.copy()						
    img = img[::-1, :, :]   				# 1 read in RGB oreder
    img = np.ascontiguousarray(img)			# 2 contiguous in memory
    #print("img.shape: ", img.shape)
    cv2.resize(img, (224, 224))
    transform = transforms.Compose([
        transforms.Lambda(lambda x: adapthist_equalize(x)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: adapthist_equalize(x)),
    transforms.ToTensor(),
])



# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir,img_no):
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
    path_cam_img = os.path.join(out_dir, "cam_pneumothorax_"+img_no+".jpg")
    cv2.imwrite(path_cam_img, cam_img)

def grad_cam(img_no,net_no,network):
#path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.3678.1517875178.953520.png' # pneumothorax
    suffix = ''
    if img_no == 1:
        suffix = '593.1517875163.595805.png'
    elif img_no == 2:
        suffix = '2447.1517875172.804337.png'
    elif img_no == 3:
        suffix = '4237.1517875181.859833.png'
    path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.' + suffix# pneumothorax 
    #path_img = '/home/ziruiqiu/comp691_DL/project/input/1.2.276.0.7230010.3.1.4.8323329.4200.1517875181.692066.png' # 0
    json_path = 'labels.json'
    output_dir = '/home/ziruiqiu/comp691_DL/project/experiments/Resnet'+net_no+'_1'
   
    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value)
               in load_json.items()}
	
	# 只取标签名
    classes = list(classes.get(key) for key in range(2))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()
    # 定义获取梯度的函数
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(module, input, output):
        fmap_block.append(output)

    # 图片读取；网络加载
    img = Image.open(path_img).convert("RGB")
    img_input = test_transforms(img)
    
    net = torch.hub.load('pytorch/vision:v0.10.0', network, pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 1)
    #pthfile = 'models/pneumothorax_experiment_VGG16_8_grad_cam_L2_1e-3_epoch20_20.pth'
    pthfile = '/home/ziruiqiu/comp691_DL/project/models/resnet/resnet'+net_no+'_1.pth' #BCE
    net.load_state_dict(torch.load(pthfile))
    net.eval()														# 8
    print(net)

    # 注册hook
    net._modules.get('layer4').register_forward_hook(farward_hook)	# 9
    net._modules.get('layer4').register_backward_hook(backward_hook)

    # forward
    img_input = img_input.unsqueeze(0)
    print("input:", img_input.shape)
    output = net(img_input)
    print("output:",output)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: ",classes[idx])

    # backward
    net.zero_grad()
    class_loss = output[0,idx]
    print("class_loss: ", class_loss)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    img_visualize = cv2.imread(path_img, 1)
    cam_show_img(img_visualize, fmap, grads_val, output_dir,str(img_no))

if __name__ == '__main__':
    network = 'resnet101'
    net_no = '101'
    img_no = 0
    for i in range(3):
        img_no += 1
        grad_cam(img_no,net_no,network)
