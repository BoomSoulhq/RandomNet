'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, us=[False, False]):
        super(USConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.width_mult = None
        self.us = us

    def forward(self, inputs):
        in_channels = inputs.shape[1] // self.groups if self.us[0] else self.in_channels // self.groups
        #print("in_channels",in_channels)
        # out_channels = make_divisible(self.out_channels * self.width_mult) if self.us[1] else self.out_channels
        out_channels = int(self.out_channels * self.width_mult) if self.us[1] else self.out_channels
        # print("out_channels",out_channels)

        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_list = None):
        super(USBatchNorm2d, self).__init__(num_features, affine=True, track_running_stats=False)
        self.width_id = None
        width_list = [0.25, 0.5, 0.75, 1]
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(self.num_features, affine=False) for _ in range(len(width_list))
        ])
        # raise NotImplementedError

    def forward(self, inputs):
        num_features = inputs.size(1)
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
    
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = USConv2d(in_channels=512, out_channels=10, kernel_size=1,
                             us=[True, False])

    def forward(self, x):
        out = self.features(x)
        #out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out.view(out.size(0), -1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [USConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, stride=1, padding=1,
                             us=[True, True]),
                           USBatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
