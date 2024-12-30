import os
import matplotlib.pyplot as plt


def read_axis_from_files(txt1_path, txt2_dir):
    """
    读取txt1中存储的文件名，并从对应的txt2文件中读取第三行轴位数据。

    Args:
        txt1_path (str): txt1文件的路径。
        txt2_dir (str): txt2文件所在的目录。

    Returns:
        List[int]: 读取到的所有轴位数据。
    """
    axis_data = []  # 存储轴位数据

    # 打开txt1文件，逐行读取对应的文件名
    with open(txt1_path, 'r', encoding='gbk') as f:
        file_names = [line.strip() + ".txt" for line in f if line.strip()]

    # 遍历所有文件名，去txt2目录中找到文件并读取第三行数据
    for file_name in file_names:
        file_path = os.path.join(txt2_dir, file_name)

        if os.path.exists(file_path):  # 检查文件是否存在
            with open(file_path, 'r', encoding='gbk') as file:
                lines = file.readlines()
                if len(lines) >= 3:  # 确保文件有第三行
                    try:
                        axis = float(lines[2].split('：')[1]) # 读取第三行并转为浮点数
                        axis_data.append(axis)
                    except ValueError:
                        print(f"Warning: 无法解析文件 {file_name} 的第三行数据")
        else:
            print(f"Warning: 文件 {file_name} 不存在于目录 {txt2_dir}")

    return axis_data


def plot_axis_distribution(axis_data):
    """
    绘制轴位数据的分布图（直方图）。

    Args:
        axis_data (List[int]): 轴位数据列表。
    """
    plt.figure(figsize=(10, 6))
    plt.hist(axis_data, bins=36, range=(0, 180), color='skyblue', edgecolor='black')
    plt.title("Axis Distribution")
    plt.xlabel("Axis Value (Degrees)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    # 设置文件路径
    txt1_path = r"D:\shishai\NIRDatasets\datasets\dataset\test.txt"  # 替换成txt1文件的路径
    txt2_dir = r"D:\shishai\NIRDatasets\datasets\dataset\labels"  # 替换成txt2文件所在的目录

    # 读取轴位数据
    axis_data = read_axis_from_files(txt1_path, txt2_dir)

    # 检查数据并绘图
    if axis_data:
        print(f"成功读取到 {len(axis_data)} 条轴位数据。")
        plot_axis_distribution(axis_data)
    else:
        print("未读取到有效的轴位数据，请检查文件路径和数据内容。")
