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
        # self.trans = torchvision.transforms.Compose([
        #                                             torchvision.transforms.ToTensor(),
        #                                             torchvision.transforms.RandomHorizontalFlip(p=0.5),
        #                                              torchvision.transforms.RandomVerticalFlip(p=0.5),
        #                                              torchvision.transforms.RandomRotation(90, expand=False, center=None, fill=0, resample=None)
        #                                              ])
        self.trans_crop = torchvision.transforms.Compose([
                                                          torchvision.transforms.RandomCrop(64)
                                                     ])
        self.trans_img = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                                                     ])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 拿到的图片
        name_image = self.name[index]

        label_path = os.path.join(self.path[:-5], 'labels',  name_image + ".txt")  # D:\shishai\Primary_school_data\data_version_2\labels\20170113_1164R.txt

        img_path = os.path.join(self.path[:-5], 'images', name_image)  # D:\shishai\Primary_school_data\data_version_2\images\20181220_1125R

        img_list = os.listdir(img_path)

        eye_img_list = []
        for i in img_list:
                eye_img_list.append(i)

        eye_img_list = eye_img_list[:1] + eye_img_list[10:] + eye_img_list[1:10]

        eye_img = []
        for i in eye_img_list:
            img = cv2.imread(os.path.join(img_path, i))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eye_img.append(img)

        eye_img = np.array(eye_img).transpose(0, 3, 1, 2).reshape(-1, img.shape[0], img.shape[0]) # (54, 80, 80)
        # eye_img = np.array(eye_img)


        # 处理一下瞳孔距离
        f = open(label_path)
        txt = []
        for line in f:
            txt.append(line.strip())

        txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1])]

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
    i = 1
    # 路径改一下
    dataset = Datasets(r"D:\shishai\Primary_school_data\data_version_2\images")
    for a, b in dataset:
        # print(i)
        print(a.shape)
        print(b)
        exit()
        # save_image(a, f"./img/{i}.jpg", nrow=1)
        # save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 2:
            break
