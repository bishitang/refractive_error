import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
# from model import ghostnet
from model_resnet import RetNet18
from model_ghostnet import ghostnet

class Tester:
    def __init__(self, model_path, model_copy_path):
        self.model_path = model_path
        self.model_copy_path = model_copy_path

        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络
        self.net = ghostnet().to(self.device)

        # 判断是否存在模型
        if os.path.exists(self.model_path):
            # 使用 map_location 将权重加载到正确的设备
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"Loaded model weights from {self.model_path}!")
        else:
            print("No Param!")

    def test(self, img, txt):
        # img: numpy array of shape [18, H, W]
        # txt: list of labels, [AX]

        # 确保 img 是一个 NumPy 数组
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected img to be a numpy array, but got {type(img)}")

        # 转换为 torch tensor
        img_tensor = torch.from_numpy(img).float().to(self.device)  # [18, 80, 80]
        img_tensor = img_tensor.unsqueeze(0)  # [1, 18, 80, 80]在张量的指定位置添加一个新的维度

        self.net.eval()
        with torch.no_grad():
            out = self.net(img_tensor)  # [1, 1]

        out_value = out[0, 0].item()
        label = txt[0]

        print(f"Predicted AX: {out_value}, Label AX: {label}")

        # 计算最小角度差值
        diff = abs(label - out_value)
        ax = min(diff, 180.0 - diff)
        print(f"Difference (ax): {ax}")

        # 分类 AX
        if ax <= 5.0:
            AX = 1
        elif ax <= 10.0:
            AX = 2
        elif ax <= 15.0:
            AX = 3
        elif ax <= 20.0:
            AX = 4
        elif ax <= 25.0:
            AX = 5
        else:
            AX = 0

        # 记录 AX 分类结果
        with open("results_AX.txt", "a") as f:
            f.write(str(AX) + '\n')

        # 判断近视变远视，远视变近视的错误的比例
        # 假设当差异大于25时，标记为错误
        if ax > 25:
            error_eye = 1
        else:
            error_eye = 0

        # 记录错误比例
        with open("results_error_eye.txt", "a") as f:
            f.write(str(error_eye) + '\n')

if __name__ == '__main__':
    PATH = r"D:\shishai\NIRDatasets\datasets\dataset/images"

    # 读取测试文件列表
    with open(r"D:\shishai\NIRDatasets\datasets\dataset\test.txt", encoding='gbk') as f:
        img_list = [line.strip() for line in f]

    # 初始化 Tester
    t = Tester(
        model_path=r'D:\shishai\model\github\refractive_error\axis_prediction\params_ghostnet_axis10'
                   r'\ghostnet_axis10_val_acc_0.509_15.604_epoch201.plt',
        model_copy_path=r'./model_{}_{}.plt'
    )

    for j, person in enumerate(img_list):
        person_path = os.path.join(PATH, person)  # 例如: D:\shishai\Primary_school_data\data_version_2\test\20170101_1152L

        label_path = person_path.replace('images', 'labels') + '.txt'
        if not os.path.exists(label_path):
            print(f"Label file not found for {person} at {label_path}. Skipping.")
            continue

        with open(label_path, encoding='gbk') as f_label:
            txt = [line.strip() for line in f_label]

        # 假设标签文件的第三行（索引2）包含 AX，格式类似于 "AX：<value>"
        try:
            label_AX = float(txt[2].split('：')[1])
        except (IndexError, ValueError) as e:
            print(f"Error reading label for {person}: {e}. Skipping.")
            continue

        # 获取 18 张眼睛图像
        eye_img_list = sorted(os.listdir(person_path), key=lambda x: int(os.path.splitext(x)[0]))  # 按照文件名排序，例如 '1.png', '2.png', ...

        if len(eye_img_list) != 18:
            print(f"Person {person} has {len(eye_img_list)} images, expected 18. Skipping.")
            continue

        eye_img = []
        for img_name in eye_img_list:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Image {img_name} not found or cannot be read for {person}. Skipping.")
                eye_img = []
                break

            img = img.astype(np.float32) / 255.0
            eye_img.append(img)

        if len(eye_img) != 18:
            print(f"Person {person} has incomplete images. Skipping.")
            continue

        # 将图像堆叠为 [18, 256, 256]
        eye_img = np.stack(eye_img, axis=0)  # [18, 256, 256]

        # 进行测试
        t.test(eye_img, [label_AX])
