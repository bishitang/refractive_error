import os


def filter_and_copy_files(txt1_path, txt2_dir, output_file_path):
    """
    筛选轴位数据在 25-140 度范围内的文件，并将文件名（去掉.txt后缀）保存到输出文件中。

    Args:
        txt1_path (str): txt1文件的路径。
        txt2_dir (str): txt2文件所在的目录。
        output_file_path (str): 输出结果文件（ax_low.txt）的路径。
    """
    valid_files = []  # 存储满足条件的文件名（不带后缀）

    # 打开txt1文件，逐行读取对应的文件名
    with open(txt1_path, 'r', encoding='gbk') as f:
        file_names = [line.strip() + ".txt" for line in f if line.strip()]  # 添加.txt后缀

    # 遍历所有文件名，检查第三行轴位数据是否在25-140之间
    for file_name in file_names:
        file_path = os.path.join(txt2_dir, file_name)

        if os.path.exists(file_path):  # 检查文件是否存在
            with open(file_path, 'r', encoding='gbk') as file:
                lines = file.readlines()
                if len(lines) >= 3:  # 确保文件有第三行
                    try:
                        axis = float(lines[2].split('：')[1])  # 读取第三行数据
                        if  15 <= axis <= 165:  # 判断轴位数据是否在范围内
                            valid_files.append(file_name[:-4])  # 去掉 ".txt" 后缀
                    except (ValueError, IndexError):
                        print(f"Warning: 无法解析文件 {file_name} 的第三行数据")
        else:
            print(f"Warning: 文件 {file_name} 不存在于目录 {txt2_dir}")

    # 将满足条件的文件名写入到输出文件（不带.txt后缀）
    with open(output_file_path, 'w', encoding='gbk') as out_file:
        for valid_file in valid_files:
            out_file.write(valid_file + '\n')

    print(f"筛选完成，共找到 {len(valid_files)} 个符合条件的文件，结果已保存到 {output_file_path}")


if __name__ == "__main__":
    # 设置文件路径
    txt1_path = r"D:\shishai\NIRDatasets\datasets\dataset\all.txt"   # 替换成txt1文件的路径
    txt2_dir = r"D:\shishai\NIRDatasets\datasets\dataset\labels"    # 替换成txt2文件所在的目录
    output_file_path = r"D:\shishai\NIRDatasets\datasets\dataset\ax.txt"  # 替换成输出文件路径

    # 执行筛选操作
    filter_and_copy_files(txt1_path, txt2_dir, output_file_path)
