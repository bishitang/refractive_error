"""
训练器模块
"""
import os
from model import MobileNetV3_Large
import torch
import datasets
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import cv2
from thop import profile
# 训练器
class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.net = MobileNetV3_Large().to(self.device)

        # 优化器，这里用的Adam，跑得快点
        # self.opt = torch.optim.Adam(self.net.parameters() ,lr = 0.00001)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.00001, weight_decay=0.0001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=500, gamma=0.1, last_epoch=-1)
        # self.loss_func = nn.SmoothL1Loss()
        self.loss_func = nn.CrossEntropyLoss()
        # 这里直接使用二分类交叉熵来训练，效果可能不那么好
        # 可以使用其他损失，比如DiceLoss、FocalLoss、BCELoss()之类的CrossEntropyLoss
        # self.loss_func = nn.MSELoss()
        # self.loss_func = MultiClassDiceLoss()
        # self.loss_func2 = MultiCEFocalLoss(3)
        # self.loss_func = DiceLoss()
        trainset_list = []
        f = open(os.path.join(path, 'train_RL.txt'), encoding='gbk')
        for line in f:
            trainset_list.append(line.strip())
        f.close()

        valset_list = []
        f = open(os.path.join(path, 'val_RL.txt'), encoding='gbk')
        for line in f:
            valset_list.append(line.strip())
        f.close()

        # 设备好，batch_size和num_workers可以给大点
        self.train_dataset = DataLoader(datasets.Datasets(trainset_list, path, True), batch_size=16, shuffle=True, num_workers=0)
        self.val_dataset = DataLoader(datasets.Datasets(valset_list, path, False), batch_size=16, shuffle=True, num_workers=0)

        # 判断是否存在模型  1`
        if os.path.exists(self.model):
            # self.net.load_state_dict(torch.load(model))
            self.net = torch.load(model).to(self.device)
            print(f"Loaded{model}!")
        else:
            print("No Param!")
        # print(self.net)
        # exit()
        # os.makedirs(img_save_path, exist_ok=True)


    # 训练
    def train(self, stop_value):
        epoch = 0
        max_val_acc = 0
        while True:
            for inputs, labels in tqdm(self.train_dataset, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.train_dataset)):
                # 图片和分割标签
                inputs, labels = inputs.float().to(self.device), labels.to(self.device)

                # 输出生成的图像
                out = self.net(inputs)

                loss = self.loss_func(out, labels[:, 0].long())

                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            print("模型损失：" + str(loss))
            if epoch % 20 == 0:
                print("网络输出视力：" + str(torch.argmax(out[0]).item()))
                print("真实视力：" + str(labels[0].item()))
                print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}")


            # 备份
            if epoch % 20 == 0:
                self.net.eval()  # 模型评估
                print('\n\n验证集验证结果：')
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    val_all = 0
                    for inputs, labels in tqdm(self.val_dataset):  # 测试模型
                        inputs, labels = inputs.float().to(self.device), labels.to(self.device)
                        # 输出生成的图像
                        out = self.net(inputs)
                        loss = self.loss_func(out, labels[:, 0].long())
                        val_loss += loss.item() * labels.size(0)

                        for j in range(len(out.tolist())):
                            val_all += 1
                            out_acc = torch.argmax(out[j], dim=0)
                            if out_acc == labels[j][0]:
                                val_acc += 1
                    val_acc /= val_all
                    print('val Loss: {:.6f},           ±0.5D Acc: {:.6f}'.format(val_loss / (len(
                        self.val_dataset)), val_acc))

                    print("网络输出视力：" + str(torch.argmax(out[0]).item()))
                    print("真实视力：" + str(labels[0].item()))

                    if max_val_acc < val_acc:
                        max_val_acc = val_acc
                    torch.save(self.net, self.model_copy.format(epoch, val_acc, loss))
                    print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1


if __name__ == '__main__':
    for i in range(4):
        # 路径改一下
        t = Trainer(r"E:\NIRDatasets\datasets\\" + str(i + 1) + "fold",
                    r'./params/' + str(i + 1) + "fold/" + 'ResNet18_5200_0.7633928571428571_23.444395065307617.plt',
                    r'./params/' + str(i + 1) + "fold/" + 'v8model_{}_{}_{}.plt',
                    img_save_path=r'D:\shishai\UNet\train_img')
        t.train(4000)
