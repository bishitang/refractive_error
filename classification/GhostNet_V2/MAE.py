import cv2
import os
import torch
import numpy as np


# 使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'./params/model_760_0.734375_0.09518831968307495.plt'
# model_760_0.734375_0.09518831968307495
# 判断是否存在模型
if os.path.exists(model_path):
    # self.net.load_state_dict(torch.load(model))
    net = torch.load(model_path).to(device)
    print(f"Loaded{model_path}!")
else:
    print("No Param!")

PATH = r"D:\shishai\Primary_school_data\data_version_3"
f=open(os.path.join(PATH, 'test.txt'), encoding='gbk')
img_list=[]
for line in f:
    img_list.append(line.strip())

# PATH = r"D:\shishai\images\data_30\TRAIN_3\31"
# img_list = os.listdir(PATH)

# print(img_list)
# exit()
person_num = 0
sph_MAE = 0
cyl_MAE = 0
ax_MAE = 0
for j, person in enumerate(img_list):
    person_path = os.path.join(PATH, 'images', person)  # D:\shishai\Primary_school_data\data_version_3\images\20191301_0513L

    label_path = person_path.replace('images', 'labels') + '.txt'
    f = open(label_path, encoding='gbk')
    txt = []
    for line in f:
        txt.append(line.strip())
    print("=================================")
    txt = [float(txt[0].split('：')[1]), float(txt[1].split('：')[1]), float(txt[2].split('：')[1])]
    print(txt)

    # if -0.5 > txt[0] and txt[0] >= -3.5:
    #     pass
    # else:
    #     continue
    #######################################################################
    # person_path = r'C:\Users\PC2021\Desktop\111'
    #######################################################################

    eye_img_list = []
    for i in range(len(os.listdir(person_path))):
        eye_img_list.append(str(i+1) + '.png')
    # eye_img_list = eye_img_list[9:] + eye_img_list[:9]
    # print(eye_img_list)
    # exit()

    eye_img = []
    for i in eye_img_list:
        img = cv2.imread(os.path.join(person_path, i))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eye_img.append(img)

    eye_img = np.array(eye_img).transpose(0, 3, 1, 2).reshape(-1, img.shape[0], img.shape[0])  # (54, 64, 64)
    eye_img = np.array(eye_img)


    eye_img = torch.from_numpy(eye_img.transpose(1,2,0)).to(device)
    eye_img = eye_img.transpose(2, 0).transpose(1, 2).unsqueeze(0).float()# torch.Size([1, 256, 256])
    net.eval()
    out = net(eye_img)
    # out[0][2] = (out[0][2] + 10) / 20 * 180
    # print(out)
    # if out[0][2] > 180:
    #     out[0][2] = 180
    # elif out[0][2] < 0:
    #     out[0][2] = 0
    out = out[0].tolist()
    print(out)
    # exit()
    person_num += 1
    sph_MAE += abs(out[0] - txt[0])
    cyl_MAE += abs(out[1] - txt[1])
    # if abs(out[2] - txt[2]) > 90:
    #     ax_MAE_ = 180 - abs(out[2] - txt[2])
    # else:
    #     ax_MAE_ = abs(out[2] - txt[2])
    # ax_MAE_ = abs(out[2] - txt[2])
    # ax_MAE += ax_MAE_
    # print(out[0].tolist())
    # exit()
print(sph_MAE)
print(cyl_MAE)
# print(ax_MAE)
print('总人数：' + str(person_num))
print(sph_MAE / person_num)
print(cyl_MAE / person_num)
# print(ax_MAE / person_num)
