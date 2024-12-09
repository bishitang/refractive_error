import torch
import torch.nn as nn
from torch.nn import functional as F


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层的步幅应始终为1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数或空间尺寸不同，需要进行下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)
        out += extra_x
        out = F.relu(out)
        return out


class RetNet18(nn.Module):
    def __init__(self):
        super(RetNet18, self).__init__()
        self.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            RestNetBasicBlock(64, 64, stride=1),
            RestNetBasicBlock(64, 64, stride=1)
        )

        self.layer2 = nn.Sequential(
            RestNetDownBlock(64, 128, [2, 1]),
            RestNetBasicBlock(128, 128, stride=1)
        )

        self.layer3 = nn.Sequential(
            RestNetDownBlock(128, 256, [2, 1]),
            RestNetBasicBlock(256, 256, stride=1)
        )

        self.layer4 = nn.Sequential(
            RestNetDownBlock(256, 512, [2, 1]),
            RestNetBasicBlock(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 输出W, H为1x1，通道数不变为512

        self.fc = nn.Linear(512, 1)  # 最后ax

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)  # 512x1x1
        out = out.view(out.size(0), -1)  # (batch_size, 512)
        out = self.fc(out)  # (batch_size, 2)
        out = torch.sigmoid(out) * 180  # 将输出限制在 [0, 180]
        return out


if __name__ == '__main__':
    x = torch.rand((3, 18, 80, 80))  # 3组18x80x80的数据
    model = RetNet18()
    y = model(x)
    print(f"Output shape: {y.shape}")  # 应该输出 torch.Size([3, 2])
    print(y)  # 查看具体输出值
