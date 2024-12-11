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
from model import ghostnet
# from model_resnet import RetNet18

class Tester:
    def __init__(self, model, model_copy):
        # self.img = img # 测试图片
        self.model = model
        self.model_copy = model_copy

        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def test(self, img, txt):
        self.img = img  # 测试图片
        self.img = torch.from_numpy(self.img.transpose(1,2,0)).to(self.device)
        self.img = self.img.transpose(2, 0).transpose(1, 2).unsqueeze(0).float()# torch.Size([1, 256, 256])

        self.net.eval()

        out  = self.net(self.img)


        out = out[0].tolist()
        print(out)



        print(abs(txt[0] - out[0]))
        if abs(txt[0] - out[0]) <= 0.125:  #0.0064102564102564
            SPH = 1
        elif abs(txt[0] - out[0]) <= 0.375:   #0.0192307692307692
            SPH = 2
        elif abs(txt[0] - out[0]) <= 0.625:  #0.0320512820512821
            SPH = 3
        elif abs(txt[0] - out[0]) <= 0.875:   #0.0448717948717949
            SPH = 4
        elif abs(txt[0] - out[0]) <= 1.125:   #0.0576923076923077
            SPH = 5
        else:
            SPH = 0


        with open("results_SPH.txt", "a") as f:
            f.write(str(SPH) + '\n')  # 自带文件关闭功能，不需要再写
        f.close()


        if abs(txt[1] - out[1]) <= 0.125:   #0.0217391304347826
            CYL = 1
        elif abs(txt[1] - out[1]) <= 0.375: #0.0652173913043478
            CYL = 2
        elif abs(txt[1] - out[1]) <= 0.625: #0.108695652173913
            CYL = 3
        elif abs(txt[1] - out[1]) <= 0.875: #0.1521739130434783
            CYL = 4
        elif abs(txt[1] - out[1]) <= 1.125: #0.1956521739130435
            CYL = 5
        else:
            CYL = 0


        with open("results_CYL.txt", "a") as f:
            f.write(str(CYL) + '\n')  # 自带文件关闭功能，不需要再写
        f.close()





        # 判断近视变远视，远视变近视的错误的比例
        if out[0] < -0.125: #0.0064102564102564
            error_eye = 1
        else:
            error_eye = 0

        with open("results_error_eye.txt", "a") as f:
            f.write(str(error_eye) + '\n')  # 自带文件关闭功能，不需要再写
        f.close()



if __name__ == '__main__':
    PATH = r"D:\shishai\NIRDatasets\datasets\dataset/images"

    f = open(r"D:\shishai\NIRDatasets\datasets\dataset\test.txt", encoding='gbk')
    # 路径改一下
    t = Tester(r'D:\shishai\model\github\refractive_error\GHost\params_v1\ghostnet_v1_val_acc_0.706_0.153_epoch748.plt', r'./model_{}_{}.plt')
    # ghostnet_val_acc_0.688_0.223_epoch77.plt
    # ghostnet_val_acc_0.670_0.219_epoch98.plt
    # ghostnet_val_acc_0.705_0.158_epoch141.plt   81.52 87.04
    # ghostnet_val_acc_0.705_0.169_epoch105.plt   82.66 84.57
    # ghostnet_val_acc_0.705_0.171_epoch100.plt   82.47 84.76


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
        # txt[0] = (txt[0] + 11.75) / 19.5
        # txt[1] = (txt[1] + 5.75) / 5.75

        eye_img_list = []
        for i in range(len(os.listdir(person_path))):
            eye_img_list.append(str(i+1) + '.png')
        # eye_img_list = eye_img_list[9:] + eye_img_list[:9]
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