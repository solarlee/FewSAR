import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from Evison import Display, show_network


def main():
    # model = os.path.join("data", "E:\LibFew-shot\dataset\MSTAR-ATLNet-Conv64F-5-1-Jun-14-2022-09-59-49\checkpoints\model_last.pth")
    # # show_network(model)
    # target_layers = [model.output]

    # model = models.alexnet(pretrained = True)
    # show_network(model)
    # target_layers = [model.features[-1]]

    # model = models.mobilenet_v3_large(pretrained=True)          #源码使用trochvision中封装好的网络模型，pretrained设置为True直接下载集成预训练好的模型参数
    # show_network(model)
    # target_layers = [model.features[-1]]        #传入最后一层卷积

    # model = models.vgg16(pretrained=True)
    # show_network(model)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   #图像预处理
    # load image
    # img_path = "E:\\LibFew-shot\\dataset\\mini-imagenet\\n0153282900000005.jpg"        #可见光图像
    img_path = "E:\\LibFew-shot\\dataset\\MSTAR\\train\\BTR_60\\232.jpeg"              #读入图片的路径
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')         #转化为RGB格式
    img = np.array(img, dtype=np.uint8)               #转化为Numpy数组格式
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)   #将图片传入指定的数据预处理当中，得到tensor
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)    #使用torch.unsqueeze()方法，增加batch维度，得到输入的Tensor

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)    #将模型、目标层、是否使用GPU传入到GradCAM方法中
    target_category = 281  # tabby, tabby cat，指定类别
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()