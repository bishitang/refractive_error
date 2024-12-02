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

from model import ghostnet
class Datasets(Dataset):

    def __init__(self, name, path, augmentation_flag):
        self.path = path
        self.name = name
        self.augmentation_flag = augmentation_flag

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
            # 修改为读取灰度图像
            img = cv2.imread(os.path.join(img_path, i), cv2.IMREAD_GRAYSCALE)
            eye_img.append(img)

        # 将图像转换为 NumPy 数组并调整形状
        eye_img = np.array(eye_img).reshape(-1, img.shape[0], img.shape[0])   # (54, 80, 80)

        # 载入轴位标签
        f = open(label_path)
        txt = []
        for line in f:
            txt.append(line.strip())

        txt = [float(txt[2].split('：')[1])]
        # txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1])]
        txt = torch.Tensor(txt)

        return eye_img, txt


if __name__ == '__main__':
    i = 1

    # 测试name_image所用 --------
    path = r'D:\shishai\NIRDatasets\datasets\dataset'
    trainset_list = []
    f = open(os.path.join(path, 'train.txt'), encoding='gbk')
    for line in f:
        trainset_list.append(line.strip())
    f.close() # -------

    # 路径改一下
    dataset = Datasets(trainset_list,r"D:\shishai\NIRDatasets\datasets\dataset",False)



    for a, b in dataset:
        # print(i)
        print(a.shape)
        print(b)

        # save_image(a, f"./img/{i}.jpg", nrow=1)
        # save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 18:
            break