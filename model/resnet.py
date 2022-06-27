# -*- coding: utf-8 -*-
# @Author: Chen Renjie
# @Date:   2021-04-10 19:49:35
# @Last Modified by:   Chen Renjie
# @Last Modified time: 2021-05-30 00:44:05

"""ResNet for CIFAR10

Change conv1 kernel size from 7 to 3
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # shortcut down sample
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, padding=0, bias=False)),
                ("bn", nn.BatchNorm2d(self.expansion*out_channels))
            ]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, padding=0, bias=False)),
                ("bn", nn.BatchNorm2d(self.expansion*out_channels))
            ]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    _C = [3, 64, 128, 256, 512]
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.hooks = {}
        self.handles = {}
        self.channels = self._C[1]
        self.layer0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=self._C[0], out_channels=self._C[1], kernel_size=3, stride=1, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(num_features=self._C[1])),
            ("relu", nn.ReLU(inplace=True))
        ]))

        self.layer1 = self.make_layer(block, self._C[1], num_blocks[0], 1)
        self.layer2 = self.make_layer(block, self._C[2], num_blocks[1], 2)
        self.layer3 = self.make_layer(block, self._C[3], num_blocks[2], 2)
        self.layer4 = self.make_layer(block, self._C[4], num_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self._C[4]*block.expansion, num_classes)

    def make_layer(self, block, out_channels, block_num, stride):
        layers = OrderedDict([("block0", block(self.channels, out_channels, stride))])
        self.channels = out_channels * block.expansion
        for i in range(block_num - 1):
            layers[f"block{i+1}"] = block(self.channels, out_channels, 1)
            self.channels = out_channels * block.expansion

        return nn.Sequential(layers)


    def forward(self, x):
        x = self.layer0(x)     # (N,  64, 32, 32)
        x = self.layer1(x)     # (N,  64, 32, 32)
        x = self.layer2(x)     # (N, 128, 16, 16)
        x = self.layer3(x)     # (N, 256,  8,  8)
        x = self.layer4(x)     # (N, 512,  4,  4)
        x = self.avgpool(x)    # (N, 512,  1,  1)
        x = self.flatten(x)    # (N, 512)
        x = self.fc(x)         # (N, 10)

        return x

def resnet10(num_classes):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes)

def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# unit test
if __name__ == "__main__":
    import time
    num_classes = 10
    x = torch.rand(64, 3, 32, 32)
    for layer in [10, 18, 34, 50, 101, 152]:
        arch = eval(f"resnet{layer}")
        model = arch(num_classes).eval()
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            ts = time.time()
            out = model(x)
            t = time.time() - ts
        
        print(f"resnet{layer} params: {params/1e6:.2f}M Infer: {t:.3f}s")