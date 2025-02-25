import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['ghost_net']

# 设定整个模型的所有BN层的衰减系数，该系数用于平滑统计的均值和方差
momentum = 0.01  # 官方默认0.1，越小，最终的统计均值和方差越接近于整体均值和方差，前提是batchsize足够大
# 用于控制更新移动平均的速度，通常在 0.1 到 0.9 之间选择。这个值决定了计算批归一化时的均值和方差的更新速度。

# 保证v可以被divisor整除,返回new_v为divisor倍数
# 通常用于确保模型的某些参数（如卷积核的大小、通道数等）符合硬件的要求，避免在计算中出现不必要的浪费
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# 定义激活函数
# inplace=True 和 inplace=False 的操作在计算结果上是相同的，但内存操作上有所不同
# inplace=True会直接修改原始张量 x，没有创建新的张量：适用于无需使用原x数据的情况
# inplace=False不会修改原始张量，而是通过创建一个新的张量来存储结果：适用于需使用原x数据的情况
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


# 定义SE模块：将重要的通道，分配多的注意力权重
# 先将所有chan（4*4*4）进行池化成1*1（1*1*4）；再经过一个结点数为初始chan的四分之一的全连接层1（1*1*1）；
# 再经过一个结点数为初始chan大小的全连接层2（1*1*4），
# 得到权重矩阵，再乘以对应通道，即赋予权重
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4,
                 **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 尺寸池化成1*1，通道不变
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True) # 全连接层1
        self.act1 = act_layer(inplace=True) # 激活函数
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True) # 全连接层2

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se) # 得到权重矩阵，再乘以对应通道，即赋予权重
        return x


# 定义基本卷积模块
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        # padding = kernel_size // 2,使得输出的尺寸不变
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs, momentum=momentum)
        self.act1 = act_layer(inplace=True) # inplace=True 表示激活操作会在原地修改输入张量，减少内存使用。

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


# 定义ghost模块

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        # primary_conv执行普通的卷积操作
        # 将原来的inp通过1*1卷积浓缩到init_channels大小
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels, momentum=momentum),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # cheap_operation: 这是一个深度卷积（Depthwise Convolution），它的作用是生成额外的通道以补充 "幽灵通道"
        # 这里使用 groups=init_channels 来实现深度卷积，即每个输入通道使用一个独立的卷积核。
        # 这个操作非常便宜，因为每个通道有自己的卷积核，不涉及通道间的权重共享。
        # dw_size 来定义卷积核大小（默认为 3）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels, momentum=momentum),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    # 通过 primary_conv(x) 得到一个特征图 x1，其通道数为 init_channels。
    # 通过 cheap_operation(x1) 得到一个深度卷积生成的特征图 x2，其通道数为 new_channels。
    # 将 x1 和 x2 沿通道维度拼接，得到一个新的张量,通道数为 init_channels + new_channels。
    # 由于拼接后的通道数是init_channels + new_channels，但是我们希望最终的输出通道数为oup，out[batch, channel, H, W]
    # 所以通过 out[:, :self.oup, :, :] 截取前 oup 个通道作为最终输出。
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# 定义ghost网络基本单元
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # 搭建ghostbottleneck
        # 先通过一个ghost模块进行扩充通道（特征提取）
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # 根据步长判断是否使用深度可分离卷积对输入特征图进行高和宽的压缩（s=1/s=2两个bottleneck）
        # 如果要进行特征图的高宽压缩，则进行逐层卷积
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs, momentum=momentum)

        # 判断是否使用注意力机制模块
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # 再次利用一个ghost模块进行通道缩放
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # 判断步长是否等1、输入通道和输出通道是否一样
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else: # 如果不一样则利用深度可分离卷积和1x1卷积调整通道数，保证主干部分和残差边部分能够进行相加
            # 深度可分离卷积确保输入特征图的空间维度保持不变。大幅降低计算量
            # 1x1 卷积通过调整通道数来保证输出特征图的通道数与主干部分一致。
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs, momentum=momentum),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs, momentum=momentum),
            )

    def forward(self, x):
        residual = x


        x = self.ghost1(x)


        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)


        if self.se is not None:
            x = self.se(x)


        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


# 搭建ghost网络模型，整个网络模型完全照搬mobilenetv3，仅仅只是更换了网络基本单元
class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=2, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # 第一层是3*3的卷积核，主要缩小尺寸为原来一半
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(18, output_channel, 3, 2, 1, bias=False)  # 输入通道数为18
        self.bn1 = nn.BatchNorm2d(output_channel, momentum=momentum)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            # k：dw卷积核大小
            # exp_size：G-bottleneck第一个1*1ghost模块将输出通道维度扩大到多少
            # c：G-bottleneck第二个1*1ghost模块将输出通道维度缩小到多少
            # se_ratio：是否使用se模块
            # s:步长
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel # 上一层output_channel为新一层的input_channel
            stages.append(nn.Sequential(*layers))

        # 在最后一个 GhostBottleneck 之后，添加了一个 1*1 卷积层，用于进一步调整通道数
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        #  1*1 卷积层添加到blocks中了
        self.blocks = nn.Sequential(*stages)

        # global_pool:将所有特征图的空间维度（高和宽）缩小为 1x1,通道数不变
        # conv_head：一个 1x1 卷积层，将通道数变为 1280
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        # self.classifier = nn.Linear(output_channel, num_classes)
        # 分为两个独立的分类器
        self.classifier_sph = nn.Linear(output_channel, 1)
        self.classifier_cyl = nn.Linear(output_channel, 1)

        self.dropout_layer = nn.Dropout(p=self.dropout) if self.dropout > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)
        # if self.dropout > 0.:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.classifier(x)
        # return x


        # 独立预测 sph 和 cyl
        sph = self.classifier_sph(x)
        cyl = self.classifier_cyl(x)

        # 使用 Tanh 激活函数将输出限制在 [-1, 1] 范围内
        sph = torch.tanh(sph)
        cyl = torch.tanh(cyl)

        # 线性映射到目标范围
        sph = sph * 9.75 - 2.0  # 映射到 [-11.75, 7.75]
        cyl = cyl * 2.875 - 2.875  # 映射到 [-5.75, 0]

        # 将 sph 和 cyl 拼接为一个张量
        x = torch.cat([sph, cyl], dim=-1)

        return x  # 返回 [sph, cyl]


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # G-bneck
        # se>0,即位使用se模块

        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == '__main__':
    model = ghostnet()
    model.eval()

    input = torch.randn(1, 18, 80, 80)  # 确保输入通道数为18
    y = model(input)
    print(y.size())  # 应输出 torch.Size([1, 2])
    print(y)  # 应输出形如 [[sph, cyl]]
