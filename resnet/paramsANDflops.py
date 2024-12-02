# import torch
# from thop import profile
#
# model_path = r'.\params\model_0_0.5625_0.18468201160430908.plt'
# model = torch.load(model_path)
# # model = resnet18()
# input = torch.randn(1, 54, 80, 80).float().cuda()
# flops, params = profile(model, inputs=(input ,))
# print('flops:{}'.format(flops))
# print('params:{}'.format(params))


import torch
from thop import profile
from model import RetNet18  # 确保导入正确的模型结构

# 定义模型路径
model_path = r'D:\shishai\model\resnet\params\resnet_val_acc_0.708_0.148_epoch609.plt'

# 构建模型实例
model = RetNet18()  # 根据实际模型结构选择正确的模型初始化
model.load_state_dict(torch.load(model_path, map_location='cuda'))
model = model.cuda().eval()  # 移动到 GPU 并设置为评估模式

# 定义输入张量
input_tensor = torch.randn(1, 54, 80, 80).float().cuda()

# 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(input_tensor,))
print(f'FLOPs: {flops}')
print(f'Params: {params}')



