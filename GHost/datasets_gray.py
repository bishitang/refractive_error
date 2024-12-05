import os
import cv2
import numpy as np
import torch


class Datasets:

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
        eye_img = np.array(eye_img).reshape(-1, img.shape[0], img.shape[0])  # (54, 80, 80)

        # 处理瞳孔距离
        with open(label_path, 'r') as f:
            txt = []
            for line in f:
                txt.append(line.strip())


        txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1])]
        txt[0] = (txt[0] + 11.75) / 19.5
        txt[1] = (txt[1] + 5.75) / 5.75
        txt = torch.Tensor(txt)

        return eye_img, txt



if __name__ == '__main__':
    path = r'D:\shishai\NIRDatasets\datasets\dataset'

    # 加载训练集列表
    trainset_list = []
    with open(os.path.join(path, 'all.txt'), encoding='gbk') as f:
        for line in f:
            trainset_list.append(line.strip())

    dataset = Datasets(trainset_list, r"D:\shishai\NIRDatasets\datasets\dataset", False)

    # 初始化最小值和最大值
    min_value_sph = float('inf')
    max_value_sph = float('-inf')
    min_value_cyl = float('inf')
    max_value_cyl = float('-inf')

    # 初始化对应的行号
    min_row_sph = None
    max_row_sph = None
    min_row_cyl = None
    max_row_cyl = None

    # 遍历所有数据
    for idx, txt in enumerate(dataset):
        # 为什么不是txt【0】，txt【1】，因为tensor数据类型为元组，
        # 只有txt[1，0]，txt[1,1]才是度数，前面数据为设备信息等tensor格式
        sph = txt[1][0].item()
        cyl = txt[1][1].item()  #

        # 计算 sph 的最小值和最大值
        if sph < min_value_sph:
            min_value_sph = sph
            min_row_sph = trainset_list[idx]
        if sph > max_value_sph:
            max_value_sph = sph
            max_row_sph = trainset_list[idx]

        # 计算 cyl 的最小值和最大值
        if cyl < min_value_cyl:
            min_value_cyl = cyl
            min_row_cyl = trainset_list[idx]
        if cyl > max_value_cyl:
            max_value_cyl = cyl
            max_row_cyl = trainset_list[idx]

    print(f"sph最小值: {min_value_sph}，对应行: {min_row_sph}")
    print(f"sph最大值: {max_value_sph}，对应行: {max_row_sph}")
    print(f"cyl最小值: {min_value_cyl}，对应行: {min_row_cyl}")
    print(f"cyl最大值: {max_value_cyl}，对应行: {max_row_cyl}")