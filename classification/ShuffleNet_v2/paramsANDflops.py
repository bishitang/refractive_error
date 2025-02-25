import torch
from thop import profile

model_path = r'.\params\model_1480_0.70703125_2.184291362762451.plt'
model = torch.load(model_path)
# model = resnet18()
input = torch.randn(1, 54, 80, 80).float().cuda()
flops, params = profile(model, inputs=(input ,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))





