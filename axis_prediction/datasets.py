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
        # self.trans_crop = torchvision.transforms.Compose([
        #                                                   torchvision.transforms.RandomCrop(64)
        #                                              ])
        # self.trans_img = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        #                                              ])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 拿到的图片
        name_image = self.name[index]

        label_path = os.path.join(self.path, 'labels',  name_image + ".txt")  # D:\shishai\Primary_school_data\data_version_2\labels\20170113_1164R.txt

        img_path = os.path.join(self.path, 'images', name_image)  # D:\shishai\Primary_school_data\data_version_2\images\20181220_1125R

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

        # if self.augmentation_flag == True:
        #     # 图像增强
        #     eyes = []
        #     trans_flag = time.time()
        #     for i in range(54):
        #         torch.manual_seed(trans_flag)
        #         eye_temp = torchvision.transforms.ToPILImage()(eye_img[i])
        #         eyes.append(np.array(self.trans_img(eye_temp)))
        #     eye_img = np.array(eyes).reshape(-1, img.shape[0], img.shape[0])


        # 旋转一定角度
        # angle = random.choice([60, 120, 180, 240, 300])
        # # print(eye_img[0].shape)
        # eyes = []
        # for i in range(54):
        #     eye_temp = torchvision.transforms.ToPILImage()(eye_img[i])
        #     eyes.append(np.array(torchvision.transforms.functional.rotate(eye_temp, angle)))
        # # print(len(eyes))  # 54
        # # print(eyes[1])
        # eyes = np.array(eyes)
        # # print(eyes.shape)  # (54, 80, 80)
        # eyes = eyes.reshape(-1, img.shape[0], img.shape[0])
        # # eyes = torch.Tensor(eyes)


        # 处理一下瞳孔距离
        f = open(label_path)
        txt = []
        for line in f:
            txt.append(line.strip())

        txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1])]


        # txt[2] = txt[2]/90 *10-5
        txt = torch.Tensor(txt)


        # temp1 = eye_img[0:3]
        # temp2 = eye_img[3:6]
        # temp3 = eye_img[6:9]
        # cv2.imshow("1", temp1.transpose(1, 2, 0))
        # cv2.imshow("2", temp2.transpose(1, 2, 0))
        # cv2.imshow("3", temp3.transpose(1, 2, 0))
        # cv2.waitKey(0)

        return eye_img, txt



if __name__ == '__main__':
    x = torch.rand((3, 54, 80, 80))  # 这个代码是为了检验resnet最终效果，3组54*80*80的数据：一次传入3组图片数据
    a = ghostnet()
    y = a(x)
    print(y.shape)
