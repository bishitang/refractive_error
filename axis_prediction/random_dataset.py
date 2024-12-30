import random
import os

# 定义地址1路径
address1 = r'D:\shishai\NIRDatasets\datasets\dataset\ax'  # 修改为实际路径

# 输入文件路径
input_file_path = os.path.join(address1, 'ax.txt')  # txt1文件的路径

# 输出文件路径
train_file_path = os.path.join(address1, 'ax_train.txt')
val_file_path = os.path.join(address1, 'ax_val.txt')
test_file_path = os.path.join(address1, 'ax_test.txt')

# 读取txt1中的所有行
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# 去除每行末尾的换行符
lines = [line.strip() for line in lines]

# 随机打乱文件名列表
random.shuffle(lines)

# 计算划分位置
total_count = len(lines)
train_count = int(0.6 * total_count)
val_count = int(0.2 * total_count)

# 划分为训练集、验证集和测试集
train_files = lines[:train_count]
val_files = lines[train_count:train_count+val_count]
test_files = lines[train_count+val_count:]

# 将每个集合的文件名写入到对应的txt文件
with open(train_file_path, 'w') as train_file:
    for file_name in train_files:
        train_file.write(file_name + '\n')

with open(val_file_path, 'w') as val_file:
    for file_name in val_files:
        val_file.write(file_name + '\n')

with open(test_file_path, 'w') as test_file:
    for file_name in test_files:
        test_file.write(file_name + '\n')

print(f"数据集已成功划分并写入到：{train_file_path}, {val_file_path}, {test_file_path}")
