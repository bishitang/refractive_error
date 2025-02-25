import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from model import ghostnet



def cal_time(model, x, iterations=50, warmup=5):
    """
    通过 CUDA Event 方法统计模型推理时间。

    Args:
        model (torch.nn.Module): 已加载的模型。
        x (torch.Tensor): 输入张量，形状应为 [B, C, H, W]。
        iterations (int, optional): 总推理次数。默认为 50。
        warmup (int, optional): 忽略的初始推理次数。默认为 5。
    """
    model.eval()
    with torch.inference_mode():
        # 实例化两个引用对象
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        time_list = []
        for i in range(iterations):
            start_event.record()
            out = model(x)
            end_event.record()
            # 同步 CUDA 事件
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
            time_list.append(elapsed_time)
            if i < warmup:
                print(f"Warmup iteration {i+1}: {elapsed_time:.5f} seconds")
        # 忽略前 'warmup' 次的推理时间
        avg_time = sum(time_list[warmup:]) / len(time_list[warmup:])
        print(f"Average inference time after {warmup} warmup iterations: {avg_time:.5f} seconds")


def load_model(model_path, device):
    """
    加载模型及其权重。

    Args:
        model_path (str): 模型权重文件的路径。
        device (torch.device): 设备（CPU 或 CUDA）。

    Returns:
        torch.nn.Module: 加载好的模型。
    """
    model = ghostnet().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}!")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model.eval()
    return model


def preprocess_image(image_path, channels=1):
    """
    读取和预处理单张图像。

    Args:
        image_path (str): 图像文件的路径。
        target_size (tuple, optional): 目标大小 (宽, 高)。默认为 (80, 80)。
        channels (int, optional): 期望的通道数。默认为 1（灰度）。

    Returns:
        torch.Tensor: 预处理后的图像张量，形状为 [C, H, W]。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    # 调整图像大小

    img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    if channels == 1:
        img = np.expand_dims(img, axis=0)  # 形状: [1, H, W]
    else:
        # 如果是灰度图像但期望 3 通道，则转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
        img = img.transpose(2, 0, 1)  # 形状: [C, H, W]
    return torch.from_numpy(img).float()


def main():
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = (r'D:\shishai\model\github\refractive_error\GHost\params_v1'
                  r'\ghostnet_v1_val_acc_0.706_0.153_epoch748.plt')
    images_dir = r"D:\shishai\NIRDatasets\datasets\dataset/images"
    test_txt_path = r"D:\shishai\NIRDatasets\datasets\dataset\test.txt"
    expected_channels = 1  # 根据模型要求调整（1 或 3）
    target_image_size = (80, 80)  # 根据模型要求调整

    # 加载模型
    try:
        model = load_model(model_path, device)
    except FileNotFoundError as e:
        print(e)
        return

    # 读取图像列表
    if not os.path.exists(test_txt_path):
        print(f"Test list file not found at {test_txt_path}")
        return

    with open(test_txt_path, 'r', encoding='gbk') as f:
        img_list = [line.strip() for line in f if line.strip()]

    # 处理每个人的图像
    for person in img_list:
        person_path = os.path.join(images_dir, person)
        if not os.path.isdir(person_path):
            print(f"Warning: {person_path} is not a directory. Skipping.")
            continue

        # 获取所有图像文件（假设为 .png 格式）
        eye_img_list = [f for f in os.listdir(person_path) if f.lower().endswith('.png')]
        if not eye_img_list:
            print(f"No PNG images found in {person_path}. Skipping.")
            continue

        eye_img_tensors = []
        for img_name in eye_img_list:
            img_path = os.path.join(person_path, img_name)
            try:
                img_tensor = preprocess_image(img_path, channels=expected_channels)
                eye_img_tensors.append(img_tensor)
            except ValueError as ve:
                print(f"Warning: {ve}. Skipping this image.")

        # 检查图像数量是否为 18
        if len(eye_img_tensors) < 18:
            print(f"Warning: {person_path} has fewer than 18 images ({len(eye_img_tensors)}). Skipping.")
            continue
        elif len(eye_img_tensors) > 18:
            print(f"Warning: {person_path} has more than 18 images ({len(eye_img_tensors)}). Using first 18 images.")
            eye_img_tensors = eye_img_tensors[:18]

        if len(eye_img_tensors) != 18:
            print(f"Warning: {person} does not have exactly 18 valid images. Skipping.")
            continue

        # 将所有图像堆叠在通道维度上
        # 每个 img_tensor 的形状为 [1, H, W]
        # 堆叠后的形状应为 [1, 18, H, W]
        img_batch = torch.stack(eye_img_tensors, dim=1).to(device)
        print(f"Processing person: {person} with img_batch shape: {img_batch.shape}")

        # 统计推理时间
        cal_time(model, img_batch)

        # 如果您需要对输出进行处理，可以在这里添加代码
        # 例如：
        # with torch.no_grad():
        #     outputs = model(img_batch)
        #     # 处理 outputs


if __name__ == '__main__':
    main()
