import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch.nn.functional as F
from torch.nn import init
# from torchsummary import summary

class hswish(nn.Module):
    def forward(self,x):
        out=x*F.relu6(x+3,inplace=True)/6
        return out


class hsigmoid(nn.Module):
    def forward(self,x):
        out=F.relu6(x+3,inplace=True)/6
        return out

#注意力机制
class SeModule(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


#线性瓶颈和反向残差结构
class Block(nn.Module):
    def __init__(self, kernel_size, in_channel, expand_size, out_channel, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        #1*1展开卷积
        self.conv1 = nn.Conv2d(in_channel, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        #3*3（或5*5）深度可分离卷积
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        #1*1投影卷积
        self.conv3 = nn.Conv2d(expand_size, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #注意力模块
        if self.se != None:
            out = self.se(out)
        #残差链接
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=7):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(54, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.num_classes = num_classes
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.conv3 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(1280)
        self.hs3 = hswish()
        self.conv4 = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 3)
        out = self.hs3(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out = out.view(-1, self.num_classes)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=7):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(54, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.num_classes=num_classes

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.conv3=nn.Conv2d(576, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(1280)
        self.hs3 = hswish()
        self.conv4 = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        #2,576,7,7
        out = self.hs2(self.bn2(self.conv2(out)))
        #print(out.detach().cpu().numpy().shape)
        #2,576,1,1
        # print(out.shape)
        out = F.avg_pool2d(out, 3)
        #print(out.detach().cpu().numpy().shape)
        #out = out.view(out.size(0), -1)
        #print(out.detach().cpu().numpy().shape)
        out = self.hs3(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        out = out.view(-1,self.num_classes)
        return out



if __name__ == '__main__':
    # summary(net, (3, 224, 224))
    x = torch.rand((3, 54, 80, 80)).to("cuda")
    a = MobileNetV3_Large().to("cuda")
    # from torchsummary import summary
    # summary(a, input_size=(54, 80, 80))
    y = a(x)
    print(y.shape)
    # torch.save(a, r"./params/test.plt")

