import cv2
import os
import model
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
from model import ghostnet

class Tester:
    def __init__(self, model, model_copy):
        # self.img = img # 测试图片
        self.model = model
        self.model_copy = model_copy

        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # 网络
        # self.net = ResNet.RestNet18().to(self.device).to(self.device)
        self.net = ghostnet().to(self.device)
        # 判断是否存在模型
        if os.path.exists(self.model):
            # 使用 map_location 将权重加载到正确的设备
            state_dict = torch.load(self.model, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"Loaded model weights from {self.model}!")
        else:
            print("No Param!")
        # os.makedirs(img_save_path, exist_ok=True)
        # 创建混淆矩阵初始化7*7，均为0
        self.fuse_matric = [[0 for i in range(7)] for i in range(7)]

    def test(self, img, txt):
        self.img = img  # 测试图片
        self.img = torch.from_numpy(self.img.transpose(1,2,0)).to(self.device)
        self.img = self.img.transpose(2, 0).transpose(1, 2).unsqueeze(0).float()# torch.Size([1, 256, 256])

        self.net.eval()
        out  = self.net(self.img)
        # out[0]:[-8,...,2,1.2],设备信息...，为tensor
        # out[0].item()将16*7取出来
        # cls为所有test的out类别
        cls = torch.argmax(out[0]).item()


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

        # 混淆矩阵对应位置+1
        self.fuse_matric[txt_SE_cls][cls] += 1
        with open("results_fuseMatric.txt", "w") as f:
            # 混淆矩阵每一行
            for matric in self.fuse_matric:
                # 混淆矩阵该行的每一列
                for i in matric:
                    f.write(str(i) + " ")  # 自带文件关闭功能，不需要再写
                f.write("\n")
        f.close()



if __name__ == '__main__':
    PATH = r"D:\shishai\NIRDatasets\datasets\dataset/images"

    f = open(r"D:\shishai\NIRDatasets\datasets\dataset\test.txt", encoding='gbk')
    # 路径改一下
    t = Tester(r'./params/ghostnet_val_acc_0.739_1.564_epoch2481.plt', r'./model_{}_{}.plt')
    # model_1500_0.8125_0.5831555128097534           0.749
    # model_1480_0.810546875_0.46440234780311584      0.7509
    # model_1420_0.80078125_0.26848670840263367      0.749

    # model_1180_0.794921875_1.030142068862915      0.73735
    # model_1320_0.791015625_0.8055082559585571     0.74708

    # model_1680_0.79296875_0.5720806121826172


    img_list = []
    for line in f:
        img_list.append(line.strip())
    for j, person in enumerate(img_list):
        person_path = os.path.join(PATH, person)  # D:\shishai\Primary_school_data\data_version_2\test\20170101_1152L

        label_path = person_path.replace('images', 'labels') + '.txt'
        f = open(label_path, encoding='gbk')
        txt = []
        for line in f:
            txt.append(line.strip())
        # print("=================================")
        txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1]), float(txt[2].split('：')[1])]
        # print(txt)

        #######################################################################
        # person_path = r'C:\Users\PC2021\Desktop\111'
        #######################################################################

        eye_img_list = []
        for i in range(len(os.listdir(person_path))):
            eye_img_list.append(str(i+1) + '.png')
        # eye_img_list = eye_img_list[9:] + eye_img_list[:9]
        # print(eye_img_list)
        # exit()

        eye_img = []
        for i in eye_img_list:
            img = cv2.imread(os.path.join(person_path, i), cv2.IMREAD_GRAYSCALE)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            eye_img.append(img)

        eye_img = np.array(eye_img).reshape(-1, img.shape[0], img.shape[0])
        # eye_img = np.array(eye_img).transpose(0, 3, 1, 2).reshape(-1, img.shape[0], img.shape[0])  # (54, 64, 64)
        eye_img = np.array(eye_img)

        # if txt[0] <= 0:
        # if txt[0] >= -3.5 and txt[0] <= 3.5:
        t.test(eye_img, txt)
        # if t == 1 :
        #     print(os.path.join(person_path, i))