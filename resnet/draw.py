import os
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.font_manager as fm

# 设置中文字体
zh_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')  # 根据实际路径修改

# 定义文件目录
directory = r"D:\shishai\NIRDatasets\datasets\labels_without_device2"

# 初始化度数统计
spherical_counts = defaultdict(int)
cylindrical_counts = defaultdict(int)
equivalent_spherical_counts = defaultdict(int)

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
            equivalent_spherical = spherical + 0.5 * cylindrical

            # 统计球镜度数、柱镜度数和等效球镜度数
            spherical_counts[spherical] += 1
            cylindrical_counts[cylindrical] += 1
            equivalent_spherical_counts[equivalent_spherical] += 1

# 将字典转换为列表，以便绘制图形
spherical_values = sorted(spherical_counts.keys())
spherical_frequencies = [spherical_counts[val] for val in spherical_values]

cylindrical_values = sorted(cylindrical_counts.keys())
cylindrical_frequencies = [cylindrical_counts[val] for val in cylindrical_values]

equivalent_spherical_values = sorted(equivalent_spherical_counts.keys())
equivalent_spherical_frequencies = [equivalent_spherical_counts[val] for val in equivalent_spherical_values]

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(spherical_values, spherical_frequencies, label="球镜度数")
plt.plot(cylindrical_values, cylindrical_frequencies, label="柱镜度数")
plt.plot(equivalent_spherical_values, equivalent_spherical_frequencies, label="等效球镜度数")
plt.xlabel("度数", fontproperties=zh_font)
plt.ylabel("人数", fontproperties=zh_font)
plt.title("总数据集球镜度数、柱镜度数和等效球镜度数分布", fontproperties=zh_font)
plt.legend(prop=zh_font)
plt.grid()
plt.show()
