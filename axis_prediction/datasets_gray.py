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

from model_ghostnet import ghostnet
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
            img = img.astype(np.float32) / 255.0
            eye_img.append(img)

        # 确保所有图像的尺寸一致
        eye_img = np.array(eye_img)
        if eye_img.ndim == 3:
            eye_img = eye_img.reshape(-1, eye_img.shape[1], eye_img.shape[2])  # (18, 80, 80)
        else:
            raise ValueError("Unexpected eye_img dimensions")

        # 处理瞳孔距离
        with open(label_path, 'r') as f:
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