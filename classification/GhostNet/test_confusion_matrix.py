import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model import ghostnet

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是 matplotlib 版本问题
    本例程使用 matplotlib-3.2.1(windows and ubuntu) 绘制正常
    需要额外安装 prettytable 库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("模型的准确率为: ", round(acc, 4))

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["类别", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
        plt.rcParams['axes.unicode_minus'] = False

        matrix = self.matrix
        plt.figure(figsize=(8,6))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        # 在图中标注数量
        thresh = matrix.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                plt.text(j, i, int(matrix[i, j]),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > thresh else "black")

        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()

class CustomDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.image_list = []
        self.label_list = []

        with open(labels_path, 'r', encoding='gbk') as f:
            for line in f:
                self.image_list.append(line.strip())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        person = self.image_list[idx]
        person_path = os.path.join(self.images_path, person)
        label_path = person_path.replace('images', 'labels') + '.txt'

        with open(label_path, 'r', encoding='gbk') as f:
            txt = f.readlines()
            txt = [float(line.strip().split('：')[1]) for line in txt[:3]]

        eye_img_list = [f"{i+1}.png" for i in range(len(os.listdir(person_path)))]
        eye_imgs = []
        for img_name in eye_img_list:
            img = cv2.imread(os.path.join(person_path, img_name), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            eye_imgs.append(img)

        eye_imgs = np.array(eye_imgs).reshape(-1, eye_imgs[0].shape[0], eye_imgs[0].shape[1])
        # eye_imgs = torch.from_numpy(eye_imgs).unsqueeze(1)  # 增加通道维度 [18,1,80,80]

        return eye_imgs, txt, person  # 返回文件夹名称

class Tester:
    def __init__(self, model_path, device):
        self.device = device
        self.net = ghostnet().to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"成功加载模型权重来自 {model_path}!")
        else:
            print("未找到模型权重文件!")
            exit()
        self.net.eval()

    def test(self, dataloader, confusion_matrix):
        misclassified_folders = []  # 用于收集符合条件的文件夹名称
        with torch.no_grad():
            for eye_imgs, txts, folder_names in tqdm(dataloader, desc="正在测试"):
                eye_imgs = eye_imgs.to(self.device)  # [batch, C, H, W]
                outputs = self.net(eye_imgs)  # [batch, num_classes]
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                # 提取 txts 中的值
                txt = [t.item() for t in txts]

                # 根据 txts 计算 txt_SE_cls
                labels = []
                if txt[1] % 0.5 == 0.25:
                    txt_SE = txt[0] + (txt[1] - 0.25) * 0.5
                elif txt[1] % 0.5 == 0.0:
                    txt_SE = txt[0] + txt[1] * 0.5
                else:
                    print("txt[1] 的值不符合预期！")
                    exit()

                if txt_SE < -6:
                    txt_SE_cls = 0
                elif -6 <= txt_SE < -3:
                    txt_SE_cls = 1
                elif -3 <= txt_SE < -0.5:
                    txt_SE_cls = 2
                elif -0.5 <= txt_SE <= 0.5:
                    txt_SE_cls = 3
                elif 0.5 < txt_SE <= 3:
                    txt_SE_cls = 4
                elif 3 < txt_SE <= 5:
                    txt_SE_cls = 5
                elif txt_SE > 5:
                    txt_SE_cls = 6
                else:
                    print("数据集类别有问题！！")
                    exit()
                labels.append(txt_SE_cls)

                confusion_matrix.update(preds, labels)

                # 检查是否为标签为“轻度近视”（索引2）且预测为“正视”（索引3）
                true_label = txt_SE_cls
                pred_label = preds[0]  # 假设 batch_size=1
                if true_label == 4 and pred_label == 3:
                    misclassified_folders.append(folder_names[0])

        # 测试完成后输出符合条件的文件夹名称
        print("\n标签为“正视”，预测为“轻度近视”的文件夹如下:")
        for folder in misclassified_folders:
            print(folder)

        # 如果需要，可以将结果保存到文件中
        # with open('misclassified_folders.txt', 'w', encoding='utf-8') as f:
        #     for folder in misclassified_folders:
        #         f.write(folder + '\n')

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 初始化混淆矩阵
    num_classes = 7
    labels = ["重度近视", "中度近视", "轻度近视", "正视", "轻度远视", "中度远视", "重度远视"]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)

    # 初始化Tester
    model_path = r'./params/ghostnet_val_acc_0.739_1.564_epoch2481.plt'
    tester = Tester(model_path, device)

    # 准备数据集
    images_path = r"D:\shishai\NIRDatasets\datasets\dataset\images"
    labels_path = r"D:\shishai\NIRDatasets\datasets\dataset\test.txt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 根据 ghostnet 的需求添加其他 transform，如需要
    ])

    dataset = CustomDataset(images_path, labels_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # 进行测试
    tester.test(dataloader, confusion)

    # 展示结果
    confusion.plot()
    confusion.summary()

if __name__ == '__main__':
    main()
