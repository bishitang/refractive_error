import os
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.font_manager as fm

# 设置中文字体
zh_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # 根据实际路径修改

# 定义文件目录
directory = r"D:\shishai\NIRDatasets\datasets\labels_without_device2"
# source_folder = r"D:\shishai\NIRDatasets\datasets\数据划分（晨哥）\图像集\images"  # 地址1：包含文件夹的源目录
# destination_folder = r"D:\shishai\NIRDatasets\datasets\4fold"  # 地址2：目标目录

# 初始化计数器
high_myopia = 0  # 重度近视
moderate_myopia = 0  # 中度近视
mild_myopia = 0  # 轻度近视
emmetropia = 0  # 正视
mild_hyperopia = 0  # 轻度远视
moderate_hyperopia = 0  # 中度远视
high_hyperopia = 0  # 重度远视

# 初始化度数统计
spherical_counts = defaultdict(int)
cylindrical_counts = defaultdict(int)

# 遍历目录中的所有txt文件
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r', encoding='gbk') as file:
            # 读取文件的前两行
            first_line = file.readline().strip()
            second_line = file.readline().strip()

            # 提取球镜度数和柱镜度数
            spherical_value = first_line.split('：')[1] if '：' in first_line else first_line.split(':')[1]
            cylindrical_value = second_line.split('：')[1] if '：' in second_line else second_line.split(':')[1]

            spherical = float(spherical_value)
            cylindrical = float(cylindrical_value)
            # 计算条件
            value = spherical + 0.5 * cylindrical
            category = None  # 用于存储类别

            # 判断条件是否在范围内并统计数量
            if value < -6.0:
                high_myopia += 1
                category = "high_myopia"
            elif -6.0 <= value < -3.0:
                moderate_myopia += 1
                category = "moderate_myopia"
            elif -3.0 <= value < -0.5:
                mild_myopia += 1
                category = "mild_myopia"
            elif -0.5 <= value <= 0.5:
                emmetropia += 1
                category = "emmetropia"
            elif 0.5 < value <= 3.0:
                mild_hyperopia += 1
                category = "mild_hyperopia"
            elif 3.0 < value <= 5.0:
                moderate_hyperopia += 1
                category = "moderate_hyperopia"
            else:
                high_hyperopia += 1
                category = "high_hyperopia"

            # 统计球镜度数和柱镜度数
            spherical_counts[spherical] += 1
            cylindrical_counts[cylindrical] += 1

            # # 复制文件夹到目标目录
            # if category:
            #     # 获取对应文件夹名称（假设txt文件名即文件夹名）
            #     folder_name = filename[:-4]  # 去掉".txt"扩展名
            #     source_path = os.path.join(source_folder, folder_name)
            #     dest_path = os.path.join(destination_folder, category)
            #     os.makedirs(dest_path, exist_ok=True)  # 如果目标目录不存在则创建
            #
            #     # 检查源文件夹是否存在
            #     if os.path.exists(source_path):
            #         dest_folder_path = os.path.join(dest_path, folder_name)
            #         if not os.path.exists(dest_folder_path):
            #             shutil.copytree(source_path, dest_folder_path)
            #             print(f"已将 {folder_name} 复制到 {category} 类别文件夹")
            #     else:
            #         print(f"源文件夹 {source_path} 不存在，跳过。")

# 输出结果
print(f"重度近视：{high_myopia}")
print(f"中度近视：{moderate_myopia}")
print(f"轻度近视：{mild_myopia}")
print(f"正视：{emmetropia}")
print(f"轻度远视：{mild_hyperopia}")
print(f"中度远视：{moderate_hyperopia}")
print(f"重度远视：{high_hyperopia}")
