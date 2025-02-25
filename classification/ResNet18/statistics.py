
fuseMatric_path = r'.\results_fuseMatric.txt'
f = open(fuseMatric_path, encoding='gbk')
fuseMatric_txt_ = []
for line in f:
    fuseMatric_txt_.append(line.strip().split(" "))

fuseMatric_txt = [[] for i in range(7)]
for i, Matric in enumerate(fuseMatric_txt_):
    for j in Matric:
        fuseMatric_txt[i].append(int(j))

all = 0
for i in range(7):
    for j in range(7):
        all += fuseMatric_txt[i][j]

acc = 0
for i in range(7):
    acc += fuseMatric_txt[i][i]


acc /= all
print("七分类准确率：")
print(all)
print(acc)
print()


macro_r = 0
num = 0
for i, Matric in enumerate(fuseMatric_txt):
    if Matric[i]:
        macro_r += Matric[i] / sum(Matric)
        num += 1
        # print(Matric[i] / sum(Matric))
macro_r /= num
print("macro_R查全率：")
print(macro_r)
print()


macro_p = 0
num = 0
for i, Matric in enumerate(fuseMatric_txt):
    macro_p_temp = fuseMatric_txt[0][i] + fuseMatric_txt[1][i] + fuseMatric_txt[2][i] + fuseMatric_txt[3][i] + fuseMatric_txt[4][i] + fuseMatric_txt[5][i] + fuseMatric_txt[6][i]
    if macro_p_temp:
        # print(macro_p_temp)
        macro_p += fuseMatric_txt[i][i] / macro_p_temp
        num += 1
macro_p /= num
print("macro_P查准率：")
print(macro_p)
print()


macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r)
print("macro_f1率：")
print(macro_f1)
print()


pe = 0
for i, Matric in enumerate(fuseMatric_txt):
    raw = fuseMatric_txt[0][i] + fuseMatric_txt[1][i] + fuseMatric_txt[2][i] + fuseMatric_txt[3][i] + fuseMatric_txt[4][i] + fuseMatric_txt[5][i] + fuseMatric_txt[6][i]
    column = sum(Matric)
    pe += raw * column
pe /= (all * all)
p0 = acc
kappa = (p0 - pe) / (1 - pe)
print("kappa率：")
print(kappa)
print()