'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class USConv2d(nn.Conv2d):  # 重构卷积层
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 us=[False, False]):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       dilation=dilation, groups=groups, bias=bias)
        self.width_mult = None
        self.us = us

    def forward(self, inputs):
        in_channels = inputs.shape[1] // self.groups if self.us[0] else self.in_channels // self.groups
        # print("in_channels",in_channels)
        # out_channels = make_divisible(self.out_channels * self.width_mult) if self.us[1] else self.out_channels
        out_channels = int(self.out_channels * self.width_mult) if self.us[1] else self.out_channels
        # print("out_channels",out_channels)

        weight = self.weight[:out_channels, :in_channels, :, :]  # 参数截取
        if self.bias is not None:  # 偏置截取与节省内存
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_list=None):
        super(USBatchNorm2d, self).__init__(num_features, affine=True, track_running_stats=False)
        self.width_id = None
        width_list = [0.25, 0.5, 0.75, 1]
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(self.num_features, affine=False) for _ in range(len(width_list))
        ])
        # raise NotImplementedError

    def forward(self, inputs):
        num_features = inputs.size(1)  # 获取输入数据的第二个维度，也就是特征数量。
        y = F.batch_norm(
            inputs,
            self.bn[self.width_id].running_mean[:num_features],
            self.bn[self.width_id].running_var[:num_features],
            self.weight[:num_features],
            self.bias[:num_features],
            self.training,
            self.momentum,
            self.eps)
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, us=[False, False]):
        super(BasicBlock, self).__init__()
        self.us = us
        self.out_channels = out_channels
        self.conv1 = USConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, us=[self.us[0], True])
        self.bn1 = USBatchNorm2d(out_channels)

        self.conv2 = USConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, us=[True, False])  # 卷积层主干路径
        self.bn2 = USBatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # 定义一个空的对象,表示它目前没有任何层,shortcut表示残差连接(直接路径)

        if stride != 1 or in_channels != self.expansion * out_channels:  # stride不等于1说明两个相加的部分的尺寸不一样需要调整
            self.shortcut = nn.Sequential(
                USConv2d(in_channels, self.expansion * out_channels,  # 调整输出通道数为self.expansion*planes使得两个相加的输出通道数一样
                         kernel_size=1, stride=stride, bias=False, us=[True, False]),  # 调整stride使得两个相加的尺寸是一样的
                nn.BatchNorm2d(self.expansion * out_channels)
                # 残差块的主干路径(卷积层)和shortcut路径(直接连接)的通道数可以不同,通过1x1卷积层来使得输出输入通道数一致
            )

    def forward(self, x):
        out_channels = int(self.out_channels * self.width_mult) if self.us[1] else self.out_channels
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = F.relu(out)[:,:out_channels, :, :]
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # residual 结构
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = USConv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False, us=[False, False])
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 除了第一个block的stride自定义,其余block的stride全部设置为1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # 注意每一个block的输出通道数都是planes * block.expansion个
        return nn.Sequential(*layers)  # *表示输入可以是任意层数的layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    # net = ResNet50()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
