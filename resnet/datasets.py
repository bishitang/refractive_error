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
        # path为根目录 ： .../dataset
        # name为提前创建好的txt文件，可为train/val/test,里面含有suda0001L、等文件夹名称信息
        # augmentation_flag 是一个标志，用于指示是否在训练模型时对数据进行增强操作
        self.path = path
        self.name = name
        self.augmentation_flag = augmentation_flag
        self.trans = torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                     torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                     ])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 拿到的图片
        # name_image为图片组名称：“20181137_1095L”
        # label_path为该图片组对应标签路径
        # img_path为图片组所在路径
        # img_list列表：listdir将img_path下所有文件转为列表
        name_image = self.name[index]
        label_path = os.path.join(self.path, 'labels',  name_image + ".txt")  # D:\shishai\Primary_school_data\data_version_2\labels\20170113_1164R.txt
        img_path = os.path.join(self.path, 'images', name_image)  # D:\shishai\Primary_school_data\data_version_2\images\20181220_1125R
        img_list = os.listdir(img_path)
        ### 这一步读入图片有问题，并不按顺序读入，需要进一步改进代码，可减少下面eye_img_list步骤

        # 将img_list值赋给eye_img_list
        eye_img_list = []
        for i in img_list:
                eye_img_list.append(i)

        # 因顺序问题重新排序
        eye_img_list = eye_img_list[:1] + eye_img_list[10:] + eye_img_list[1:10]

        #定义eye_img列表，里面存储每组图片的np格式的数据
        eye_img = []
        for i in eye_img_list:
            img = cv2.imread(os.path.join(img_path, i))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            eye_img.append(img)

        # 原先的eye_img为18组80*80*3（H,W,C）的数据列表
        # np.array将eye_img从列表转数组
        # transpose（0，3，1，2）将（N,H,W,C）转为（N,C,H,W）,其中N为N张图片
        # reshape实现将（18，3，80，80）--->（54，80，80），实现将3通道转为单通道
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

        # 打开图片组所对应标签文件，记录SPH和CYL的标签数据
        f = open(label_path)
        txt = []
        for line in f:
            txt.append(line.strip())
        # txt = ['球镜度数：-2.25'，'柱镜度数：-0.25‘，'轴位：143'，’裸眼视力：4.4‘]
        txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1])]
        # txt = [-2.25,-0.25]

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
