SPH_path = r'.\results_SPH.txt'
f = open(SPH_path, encoding='gbk')
SPH_txt = []
for line in f:
    SPH_txt.append(line.strip())
f.close()

CYL_path = r'.\results_CYL.txt'
f = open(CYL_path, encoding='gbk')
CYL_txt = []
for line in f:
    CYL_txt.append(line.strip())
f.close()

error_eye_path = r'.\results_error_eye.txt'
f = open(error_eye_path, encoding='gbk')
error_eye_txt = []
for line in f:
    error_eye_txt.append(line.strip())
f.close()

SPH_result = [0, 0, 0, 0, 0, 0]
for i in SPH_txt:
    if i == '1':
        SPH_result[0] += 1
    elif i == '2':
        SPH_result[1] += 1
    elif i == '3':
        SPH_result[2] += 1
    elif i == '4':
        SPH_result[3] += 1
    elif i == '5':
        SPH_result[4] += 1
    else:
        SPH_result[5] += 1
# 各个误差范围误的个数 0，25，50，75，100
print(SPH_result)

CYL_result = [0, 0, 0, 0, 0, 0]
for i in CYL_txt:
    if i == '1':
        CYL_result[0] += 1
    elif i == '2':
        CYL_result[1] += 1
    elif i == '3':
        CYL_result[2] += 1
    elif i == '4':
        CYL_result[3] += 1
    elif i == '5':
        CYL_result[4] += 1
    else:
        CYL_result[5] += 1

print(CYL_result)

error_eye_result = [0, 0]
for i in error_eye_txt:
    if i == '1':
        error_eye_result[0] += 1
    else:
        error_eye_result[1] += 1

print(error_eye_result)

all_eye_num = error_eye_result[0] + error_eye_result[1]

SPH_persent = []
temp = 0
for i in range(len(SPH_result) - 1):
    temp += SPH_result[i]
    SPH_persent.append(temp / all_eye_num * 100)

SPH_persent.append(100 - SPH_persent[-1])
print("球镜度数误差比例：")
print(SPH_persent)

CYL_persent = []
temp = 0
for i in range(len(CYL_result) - 1):
    temp += CYL_result[i]
    CYL_persent.append(temp / all_eye_num * 100)
CYL_persent.append(100 - CYL_persent[-1])
print("柱镜度数误差比例：")
print(CYL_persent)


# 清空文件内容
open(SPH_path, 'w').close()
open(CYL_path, 'w').close()
open(error_eye_path, 'w').close()