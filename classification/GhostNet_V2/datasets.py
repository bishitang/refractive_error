import os
import cv2
import torchvision
import numpy as np
import random
import torch
import imutils
import time
from torch.utils.data import Dataset
from torchvision.utils import save_image



# 简单的数据集，没有进行数据增强
class Datasets(Dataset):

    def __init__(self, name, path, augmentation_flag):
        # 做标签
        self.path = path
        self.name = name
        self.augmentation_flag = augmentation_flag

        self.trans_crop = torchvision.transforms.Compose([
                                                          torchvision.transforms.RandomCrop(64)
                                                     ])
        # 定义了一个包含颜色抖动（亮度、对比度、饱和度、色调）的转换
        self.trans_img = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                                                     ])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 拿到的图片
        name_image = self.name[index]

        label_path = os.path.join(self.path, 'labels', name_image + ".txt")
        img_path = os.path.join(self.path, 'images', name_image)
        img_list = os.listdir(img_path)

        eye_img_list = []
        for i in img_list:
                eye_img_list.append(i)

        eye_img_list = eye_img_list[:1] + eye_img_list[10:] + eye_img_list[1:10]

        eye_img = []
        for i in eye_img_list:
            # 读取灰度图像
            img = cv2.imread(os.path.join(img_path, i), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {os.path.join(img_path, i)}")
            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0
            eye_img.append(img)

        # 确保所有图像的尺寸一致
        eye_img = np.array(eye_img)
        if eye_img.ndim == 3:
            eye_img = eye_img.reshape(-1, eye_img.shape[1], eye_img.shape[2])  # (18, 80, 80)
        else:
            raise ValueError("Unexpected eye_img dimensions")

        with open(label_path, 'r') as f:
            txt = []
            for line in f:
                txt.append(line.strip())

        txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1])]
        txt = torch.Tensor(txt)

        if txt[1] % 0.5 == 0.25:
            txt_SE = txt[0] + (txt[1] - 0.25) * 0.5
        elif txt[1] % 0.5 == 0.0:
            txt_SE = txt[0] + txt[1] * 0.5

        if txt_SE < -6:
            txt_SE_cls = 0
        elif txt_SE >= -6 and txt_SE < -3:
            txt_SE_cls = 1
        elif txt_SE >= -3 and txt_SE < -0.5:
            txt_SE_cls = 2
        elif txt_SE >= -0.5 and txt_SE <= 0.5:
            txt_SE_cls = 3
        elif txt_SE > 0.5 and txt_SE <= 3:
            txt_SE_cls = 4
        elif txt_SE > 3 and txt_SE <= 5:
            txt_SE_cls = 5
        elif txt_SE > 5:
            txt_SE_cls = 6
        else:
            print("dataset数据集类别有问题！！")
            exit()



        # txt[2] = txt[2]/90 *10-5
        label = torch.Tensor([txt_SE_cls])


        return eye_img, label



if __name__ == '__main__':
    path = r'D:\shishai\NIRDatasets\datasets\dataset'

    # 加载训练集列表
    trainset_list = []
    with open(os.path.join(path, 'all.txt'), encoding='gbk') as f:
        for line in f:
            trainset_list.append(line.strip())

    dataset = Datasets(trainset_list, r"D:\shishai\NIRDatasets\datasets\dataset", False)